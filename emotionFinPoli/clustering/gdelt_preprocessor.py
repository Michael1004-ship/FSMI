from datetime import datetime
import os
import json
import time
import logging
import sys
import psutil
from io import BytesIO
from typing import List, Dict, Tuple
from tqdm import tqdm

import newspaper
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage

# ----------------------------
# 설정
# ----------------------------
BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
GDELT_PREFIX = "news/gdelt/"
SLEEP_SECONDS = 2.5
TARGET_DATE = datetime.utcnow().strftime("%Y-%m-%d")  # UTC 기준 오늘
SAVE_FILENAME = "news_anxiety_index.csv"

# ----------------------------
# 로깅 설정
# ----------------------------
# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/gdelt_preprocessor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_preprocessor")

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ----------------------------
# GCS 클라이언트
# ----------------------------
client = storage.Client()
bucket_raw = client.bucket(BUCKET_RAW)
bucket_index = client.bucket(BUCKET_INDEX)

def get_memory_usage() -> float:
    """현재 프로세스의 메모리 사용량을 MB 단위로 반환"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB 단위로 변환

def fetch_article_urls_from_accumulated_json(date: str) -> List[str]:
    """GCS에서 accumulated.json 파일을 읽어 URL 목록 추출"""
    start_time = time.time()
    logger.info(f"🔍 URL 수집 시작 (날짜: {date})")
    
    # 메모리 사용량 기록
    initial_memory = get_memory_usage()
    logger.debug(f"📊 초기 메모리 사용량: {initial_memory:.2f} MB")
    
    blob_path = f"{GDELT_PREFIX}{date}/accumulated.json"
    blob = bucket_raw.blob(blob_path)
    
    if not blob.exists():
        logger.error(f"❌ 파일 없음: gs://{BUCKET_RAW}/{blob_path}")
        raise FileNotFoundError(f"❌ 파일 없음: gs://{BUCKET_RAW}/{blob_path}")
    
    # 파일 크기 확인
    try:
        blob.reload()
        file_size_kb = blob.size / 1024 if blob.size is not None else 0
        logger.info(f"📦 파일 크기: {file_size_kb:.2f} KB")
    except Exception as e:
        logger.warning(f"⚠️ 파일 크기 확인 오류: {e}")
        file_size_kb = 0
    
    # 다운로드 시간 측정
    download_start = time.time()
    logger.info(f"⏳ 파일 다운로드 중...")
    content = blob.download_as_bytes()
    download_time = time.time() - download_start
    logger.info(f"⏱️ 다운로드 완료: {download_time:.2f}초 ({file_size_kb/download_time:.2f} KB/초)")
    
    # JSON 파싱
    parse_start = time.time()
    items = json.load(BytesIO(content))
    parse_time = time.time() - parse_start
    logger.info(f"📊 JSON 파싱 완료: {parse_time:.2f}초, {len(items)}개 항목")
    
    # URL 추출
    urls = [item["DocumentIdentifier"] for item in items if "DocumentIdentifier" in item]
    
    # 메모리 사용량 확인
    current_memory = get_memory_usage()
    logger.debug(f"📊 URL 추출 후 메모리: {current_memory:.2f} MB (증가: {current_memory - initial_memory:.2f} MB)")
    
    # 실행 시간 계산
    total_time = time.time() - start_time
    logger.info(f"🔗 총 {len(urls)}개의 기사 URL 수집됨 (소요 시간: {total_time:.2f}초)")
    
    return urls

def fetch_article_text(url: str) -> str:
    """신문 기사 URL에서 본문 텍스트 추출"""
    try:
        config = newspaper.Config()
        config.browser_user_agent = "Mozilla/5.0"
        article = Article(url, config=config)
        
        # 다운로드 시간 측정
        download_start = time.time()
        article.download()
        download_time = time.time() - download_start
        
        # 파싱 시간 측정
        parse_start = time.time()
        article.parse()
        parse_time = time.time() - parse_start
        
        text = article.text.strip()
        logger.debug(f"✓ 크롤링 성공: 다운로드 {download_time:.2f}초, 파싱 {parse_time:.2f}초, {len(text)} 글자")
        return text
    except Exception as e:
        logger.warning(f"❌ 크롤링 실패: {url} - {e}")
        return None

def process_articles(urls: List[str]) -> Tuple[List[str], List[str], Dict]:
    """URL 목록에서 기사 본문 추출 및 처리 통계 생성"""
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    logger.info(f"🔄 기사 크롤링 시작 (총 {len(urls)}개 URL)")
    
    texts = []
    failed_urls = []
    stats = {
        "total": len(urls),
        "success": 0,
        "failed": 0,
        "empty": 0,
        "total_chars": 0,
        "avg_chars_per_article": 0
    }
    
    # tqdm으로 진행률 표시
    for i, url in enumerate(tqdm(urls, desc="기사 처리")):
        # 100개마다 진행 상황 로깅
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - start_time
            progress = i / len(urls) * 100
            urls_per_second = i / elapsed
            estimated_total = elapsed / i * len(urls)
            remaining = max(0, estimated_total - elapsed)
            
            logger.info(f"  → {i}/{len(urls)} 처리 중 ({progress:.1f}%) - "
                       f"속도: {urls_per_second:.1f}개/초, 남은 시간: {remaining/60:.1f}분")
            logger.info(f"  → 성공: {stats['success']}개, 실패: {stats['failed']}개")
        
        logger.debug(f"[{i+1}/{len(urls)}] URL 처리 중: {url}")
        
        text = fetch_article_text(url)
        time.sleep(SLEEP_SECONDS)
        
        if text:
            if len(text) > 0:
                texts.append(text)
                stats["success"] += 1
                stats["total_chars"] += len(text)
            else:
                stats["empty"] += 1
                failed_urls.append(f"{url} (빈 텍스트)")
        else:
            stats["failed"] += 1
            failed_urls.append(url)
    
    # 통계 마무리
    process_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    if stats["success"] > 0:
        stats["avg_chars_per_article"] = stats["total_chars"] / stats["success"]
    
    logger.info(f"⏱️ 크롤링 완료: {process_time:.2f}초 (평균 {process_time/len(urls):.2f}초/URL)")
    logger.info(f"📊 메모리 사용량: {final_memory:.2f} MB (증가: {final_memory - initial_memory:.2f} MB)")
    
    return texts, failed_urls, stats

def save_texts_to_gcs(texts: List[str], date: str) -> float:
    """추출한 텍스트를 GCS에 저장"""
    start_time = time.time()
    logger.info(f"💾 GCS 저장 시작: {len(texts)}개 텍스트")
    
    # JSON 변환 시간 측정
    json_start = time.time()
    json_content = json.dumps(texts, ensure_ascii=False, indent=2)
    json_time = time.time() - json_start
    
    content_size_kb = len(json_content) / 1024
    logger.info(f"⏱️ JSON 변환: {json_time:.2f}초, 크기: {content_size_kb:.2f} KB")
    
    # GCS 업로드
    upload_start = time.time()
    output_path = f"{GDELT_PREFIX}{date}/news_text.json"
    blob = bucket_raw.blob(output_path)
    
    blob.upload_from_string(json_content, content_type='application/json')
    
    upload_time = time.time() - upload_start
    total_time = time.time() - start_time
    
    logger.info(f"⏱️ 업로드 완료: {upload_time:.2f}초 ({content_size_kb/upload_time:.1f} KB/초)")
    logger.info(f"⏱️ 총 저장 시간: {total_time:.2f}초")
    logger.info(f"📤 저장 완료: gs://{BUCKET_RAW}/{output_path} ({len(texts)}개 텍스트)")
    
    return content_size_kb

def save_failed_urls(failed_urls: List[str], date: str):
    """실패한 URL 목록을 텍스트 파일로 저장"""
    if not failed_urls:
        logger.info("👍 실패한 URL 없음")
        return
    
    output_path = f"{GDELT_PREFIX}{date}/failed_urls.txt"
    blob = bucket_raw.blob(output_path)
    
    content = "\n".join(failed_urls)
    blob.upload_from_string(content, content_type='text/plain')
    
    logger.warning(f"⚠️ 실패 URL 목록 저장: gs://{BUCKET_RAW}/{output_path} ({len(failed_urls)}개)")

def analyze_articles(urls: List[str]) -> Tuple[List, List]:
    results = []
    failed_urls = []
    details = []

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{len(urls)}] URL 처리 중: {url}")
        text = fetch_article_text(url)
        time.sleep(SLEEP_SECONDS)

        if text:
            try:
                result = finbert(text[:512])[0]
                label = result["label"].lower()
                score = result["score"]
                if label == "negative":
                    results.append(score)
                details.append([url, label, score])
            except Exception as e:
                logger.error(f"FinBERT 오류: {e}")
                failed_urls.append(url)
        else:
            failed_urls.append(url)

    return details, failed_urls

def save_results_to_gcs(details: List, failed_urls: List[str], date: str):
    total = len(details)
    negatives = [r for r in details if r[1] == "negative"]
    neg_ratio = len(negatives) / total if total > 0 else 0
    avg_neg_score = sum(r[2] for r in negatives) / len(negatives) if negatives else 0
    anxiety_index = neg_ratio * avg_neg_score

    temp_csv = "/tmp/news_anxiety_index.csv"
    with open(temp_csv, "w", encoding="utf-8") as f:
        f.write("negative_ratio,average_negative_score,anxiety_index\n")
        f.write(f"{neg_ratio:.3f},{avg_neg_score:.3f},{anxiety_index:.3f}\n\n")
        f.write("url,label,score\n")
        for row in details:
            f.write(f"{row[0]},{row[1]},{row[2]:.3f}\n")

    blob = bucket_index.blob(f"news/{date}/{SAVE_FILENAME}")
    blob.upload_from_filename(temp_csv)
    logger.info(f"✅ 지수 저장 완료 → gs://{BUCKET_INDEX}/news/{date}/{SAVE_FILENAME}")

    if failed_urls:
        temp_log = "/tmp/failed_urls.txt"
        with open(temp_log, "w") as f:
            for url in failed_urls:
                f.write(url + "\n")
        log_blob = bucket_index.blob(f"news/{date}/failed_urls.txt")
        log_blob.upload_from_filename(temp_log)
        logger.warning(f"⚠️ 실패 URL 저장 완료 → gs://{BUCKET_INDEX}/news/{date}/failed_urls.txt")

if __name__ == "__main__":
    try:
        # 전체 시작 시간 및 메모리 사용량
        total_start_time = time.time()
        initial_memory = get_memory_usage()
        
        logger.info(f"🚀 GDELT 뉴스 전처리 시작 (날짜: {TARGET_DATE})")
        logger.info(f"💻 시스템 정보: CPU {psutil.cpu_percent()}%, 메모리 {psutil.virtual_memory().percent}%")
        
        # 1. URL 수집
        urls = fetch_article_urls_from_accumulated_json(TARGET_DATE)
        
        # 2. 기사 처리
        texts, failed_urls, stats = process_articles(urls)
        
        # 3. 결과 저장
        content_size_kb = save_texts_to_gcs(texts, TARGET_DATE)
        save_failed_urls(failed_urls, TARGET_DATE)
        
        # 4. 최종 결과 요약
        total_time = time.time() - total_start_time
        final_memory = get_memory_usage()
        memory_diff = final_memory - initial_memory
        
        logger.info("=" * 60)
        logger.info(f"🎉 GDELT 뉴스 전처리 완료!")
        logger.info(f"⏱️ 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        logger.info(f"📊 메모리 사용량: {final_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
        logger.info(f"📈 처리 결과:")
        logger.info(f"  • 처리된 URL: {stats['total']}개")
        logger.info(f"  • 성공: {stats['success']}개 ({stats['success']/stats['total']*100:.1f}%)")
        logger.info(f"  • 실패: {stats['failed']}개 ({stats['failed']/stats['total']*100:.1f}%)")
        logger.info(f"  • 빈 텍스트: {stats['empty']}개")
        logger.info(f"  • 총 텍스트 크기: {content_size_kb:.2f} KB")
        if stats['success'] > 0:
            logger.info(f"  • 평균 텍스트 길이: {stats['avg_chars_per_article']:.1f}자")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"❌ 프로그램 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)
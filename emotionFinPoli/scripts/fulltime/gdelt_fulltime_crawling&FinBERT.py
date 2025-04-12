import argparse
from datetime import datetime, timedelta
import json
import os
from io import BytesIO
from newspaper import Article
import newspaper
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage
import logging
import sys
import concurrent.futures
from tqdm import tqdm
import time
import random

# ----------------------------
# 설정
# ----------------------------
BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
MODEL_NAME = "yiyanghkust/finbert-tone"
LOCAL_TEMP_DIR = "/tmp"
GDELT_PREFIX = "news/gdelt/"

# 병렬 처리 설정
MAX_WORKERS = 8  # 동시 실행 스레드 수 
MAX_RETRIES = 3  # 크롤링 재시도 횟수
CRAWL_DELAY = 0.5  # 크롤링 요청 간 지연 시간(초)
REQUEST_TIMEOUT = 10  # 요청 타임아웃(초)

# ----------------------------
# 로그 설정
# ----------------------------
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"
os.makedirs(LOG_DATE_DIR, exist_ok=True)

# 스크립트 이름 기반으로 로그 파일 설정
script_name = os.path.basename(__file__)
log_file = f"{LOG_DATE_DIR}/{script_name.replace('.py', '.log')}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_finbert")

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------
logger.info("🧠 FinBERT 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
logger.info("✅ 모델 로딩 완료")

# ----------------------------
# GCS 연결
# ----------------------------
client = storage.Client()
bucket_raw = client.bucket(BUCKET_RAW)
bucket_index = storage.Client().bucket(BUCKET_INDEX)

# ----------------------------
# 본문 크롤링 함수
# ----------------------------
def fetch_article_text(url, retry=0):
    """기사 본문 크롤링 함수 (재시도 로직 포함)"""
    if retry > MAX_RETRIES:
        return None
        
    try:
        # 크롤링 간 약간의 랜덤 지연 추가
        time.sleep(CRAWL_DELAY + random.uniform(0, 1))
        
        config = newspaper.Config()
        config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        config.request_timeout = REQUEST_TIMEOUT
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        text = article.text.strip()
        
        if not text:
            return None
            
        return text
    except Exception as e:
        # 재시도 로직
        if retry < MAX_RETRIES:
            time.sleep(retry * 2)  # 재시도마다 대기 시간 증가
            return fetch_article_text(url, retry + 1)
        else:
            return None

# ----------------------------
# 기사 처리 함수 (병렬 처리용)
# ----------------------------
def process_article(item):
    """단일 기사 크롤링 및 감정 분석 (병렬 처리용)"""
    try:
        url = item.get("DocumentIdentifier")
        if not url:
            return {
                "success": False,
                "error": "URL 없음",
                "url": None
            }
            
        # 기사 크롤링
        text = fetch_article_text(url)
        if not text:
            return {
                "success": False,
                "error": "크롤링 실패",
                "url": url
            }
            
        # 감정 분석
        result = finbert(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        
        return {
            "success": True,
            "url": url,
            "label": label,
            "score": score,
            "is_negative": label == "negative"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url if 'url' in locals() else None
        }

def get_date_range(start_date, end_date):
    """시작일과 종료일 사이의 모든 날짜를 생성"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return date_list

# ----------------------------
# 메인 함수
# ----------------------------
def main(start_date, end_date, debug=False, workers=MAX_WORKERS):
    """메인 실행 함수"""
    date_list = get_date_range(start_date, end_date)
    logger.info(f"처리할 날짜: {date_list} ({len(date_list)}일)")
    logger.info(f"병렬 처리 스레드 수: {workers}")
    
    # 전체 실행 시간 측정
    total_start_time = time.time()
    total_articles = 0
    total_processed = 0
    total_failed = 0
    
    for date_str in date_list:
        date_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"📅 {date_str} 처리 시작")
        
        # 해당 날짜의 JSON 파일 찾기 (수정된 경로)
        prefix_path = f"{GDELT_PREFIX}full/{start_date}_to_{end_date}.json"
        blob = bucket_raw.blob(prefix_path)
        
        if not blob.exists():
            logger.warning(f"🚫 {prefix_path} 데이터 파일이 없습니다.")
            # 다른 형식의 경로도 시도
            alt_path = f"{GDELT_PREFIX}{date_str}/accumulated.json"
            blob = bucket_raw.blob(alt_path)
            
            if not blob.exists():
                logger.warning(f"🚫 {date_str} 날짜의 데이터가 없습니다.")
                continue
        
        try:
            # 데이터 로드
            content = blob.download_as_bytes()
            items = json.load(BytesIO(content))
            
            if not items:
                logger.warning(f"⚠️ {date_str}에 처리할 뉴스가 없습니다.")
                continue
                
            # 특정 날짜 필터링 (필요한 경우)
            if start_date != end_date:
                filtered_items = []
                for item in items:
                    if 'DATE' in item:
                        item_date = item['DATE'][:8]  # YYYYMMDDHHMMSS 형식에서 YYYYMMDD 추출
                        item_date_formatted = f"{item_date[:4]}-{item_date[4:6]}-{item_date[6:8]}"
                        if item_date_formatted == date_str:
                            filtered_items.append(item)
                items = filtered_items
                
            total_articles += len(items)
            logger.info(f"📰 {date_str} - 총 {len(items)}개 기사 분석 시작")
            
            results = []
            details = []
            failed_urls = []
            
            # 병렬 처리를 위한 tqdm 진행 표시줄
            with tqdm(total=len(items), desc=f"{date_str} 분석 중", unit="기사") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    # 작업 제출
                    future_to_item = {executor.submit(process_article, item): item for item in items}
                    
                    # 결과 처리
                    for future in concurrent.futures.as_completed(future_to_item):
                        result = future.result()
                        pbar.update(1)
                        
                        if result["success"]:
                            total_processed += 1
                            details.append([result["url"], result["label"], result["score"]])
                            
                            if result["is_negative"]:
                                results.append(result["score"])
                        else:
                            total_failed += 1
                            failed_urls.append({
                                "url": result["url"],
                                "error": result["error"]
                            })
                        
                        # 주기적 상태 보고 (10%마다)
                        if pbar.n % max(1, len(items) // 10) == 0 or pbar.n == len(items):
                            elapsed = time.time() - date_start_time
                            progress = pbar.n / len(items) * 100
                            remaining = (elapsed / pbar.n) * (len(items) - pbar.n) if pbar.n > 0 else 0
                            logger.info(f"🔄 진행률: {progress:.1f}% ({pbar.n}/{len(items)}) - "
                                      f"성공: {len(details)}, 실패: {len(failed_urls)} - "
                                      f"예상 남은 시간: {remaining/60:.1f}분")
            
            # 결과 계산
            total_count = len(details)
            negative_ratio = len(results) / total_count if total_count > 0 else 0
            avg_negative_score = sum(results) / len(results) if results else 0
            anxiety_index = negative_ratio * avg_negative_score
            
            logger.info(f"📊 {date_str} 분석 결과:")
            logger.info(f"  • 총 기사: {len(items)}개")
            logger.info(f"  • 성공적으로 처리된 기사: {total_count}개")
            logger.info(f"  • 부정 기사 비율: {negative_ratio:.3f}")
            logger.info(f"  • 평균 부정 점수: {avg_negative_score:.3f}")
            logger.info(f"  • 불안 지수: {anxiety_index:.3f}")
            
            # 파일 저장
            save_folder = f"news/{date_str}"
            save_filename = "news_anxiety_index.csv"
            local_path = os.path.join(LOCAL_TEMP_DIR, f"{date_str}_{save_filename}")
            
            with open(local_path, "w", encoding="utf-8") as f:
                f.write("negative_ratio,average_negative_score,anxiety_index\n")
                f.write(f"{negative_ratio:.3f},{avg_negative_score:.3f},{anxiety_index:.3f}\n\n")
                f.write("url,label,score\n")
                for row in details:
                    f.write(f"{row[0]},{row[1]},{row[2]:.3f}\n")
            
            # GCS에 업로드
            blob = bucket_index.blob(f"{save_folder}/{save_filename}")
            blob.upload_from_filename(local_path)
            logger.info(f"✅ 저장 완료 → gs://{BUCKET_INDEX}/{save_folder}/{save_filename}")
            
            # 실패 URL 로그 (필요한 경우)
            if failed_urls:
                fail_log_path = f"{LOG_DATE_DIR}/gdelt_finbert_failed_{date_str}.json"
                with open(fail_log_path, 'w') as f:
                    json.dump(failed_urls, f, indent=2)
                logger.info(f"📝 실패 URL 저장됨: {fail_log_path}")
            
            # 날짜별 처리 시간
            date_elapsed_time = time.time() - date_start_time
            logger.info(f"⏱️ {date_str} 처리 완료: {date_elapsed_time:.1f}초 ({date_elapsed_time/60:.1f}분)")
            
        except Exception as e:
            logger.error(f"❌ {date_str} 처리 중 오류 발생: {str(e)}")
            if debug:
                logger.exception("상세 오류:")
            continue
    
    # 전체 실행 결과 요약
    total_elapsed_time = time.time() - total_start_time
    logger.info("\n" + "="*50)
    logger.info("📊 전체 실행 결과")
    logger.info(f"• 처리 날짜: {len(date_list)}일")
    logger.info(f"• 총 기사 수: {total_articles}개")
    logger.info(f"• 성공적으로 처리된 기사: {total_processed}개")
    logger.info(f"• 실패한 기사: {total_failed}개")
    logger.info(f"• 성공률: {(total_processed/total_articles)*100:.1f}% (성공/전체)")
    logger.info(f"• 총 소요 시간: {total_elapsed_time:.1f}초 ({total_elapsed_time/60:.1f}분)")
    logger.info("="*50)

# ----------------------------
# CLI 인터페이스
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT 뉴스 감정 분석기 (병렬 처리)")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"병렬 처리 스레드 수 (기본값: {MAX_WORKERS})")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("🔧 디버그 모드 활성화됨")
    
    try:
        main(args.start, args.end, args.debug, args.workers)
    except Exception as e:
        if args.debug:
            logger.exception("❌ 치명적 오류 발생:")
        else:
            logger.error(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)

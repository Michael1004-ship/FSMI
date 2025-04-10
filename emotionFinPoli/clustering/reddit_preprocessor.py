import os
import json
import datetime
import time
import sys
import psutil
import logging
from google.cloud import storage
from tqdm import tqdm

# ✅ 로깅 설정
from datetime import datetime

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
        logging.FileHandler(f"{LOG_DATE_DIR}/reddit_preprocessor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_preprocessor")

# ✅ 설정
BUCKET_NAME = "emotion-raw-data"
SUBREDDITS = [
    "anxiety", "depression", "dividends", "EconMonitor",
    "economics", "economy", "finance", "financialindependence",
    "investing", "MacroEconomics", "offmychest", "personalfinance",
    "StockMarket", "stocks", "wallstreetbets"
]

# ✅ 오늘 날짜 (UTC 기준 → 한국 기준이면 +9 설정)
today = datetime.utcnow().strftime("%Y-%m-%d")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def preprocess_reddit_from_gcs(bucket_name, gcs_path):
    start_time = time.time()
    logger.info(f"🔄 파일 다운로드 시작: gs://{bucket_name}/{gcs_path}")
    
    # 메모리 사용량 기록
    initial_memory = get_memory_usage()
    logger.debug(f"📊 초기 메모리: {initial_memory:.2f} MB")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise FileNotFoundError(f"❌ 파일 없음: gs://{bucket_name}/{gcs_path}")
    
    # 파일 크기 확인 (예외 처리 추가)
    try:
        blob.reload()  # 메타데이터 로드
        file_size_kb = blob.size / 1024 if blob.size is not None else 0
        logger.info(f"📦 파일 크기: {file_size_kb:.2f} KB")
    except Exception as e:
        logger.warning(f"⚠️ 파일 크기 확인 오류: {e}")
        file_size_kb = 0
    
    # 다운로드 시간 측정
    download_start = time.time()
    raw_content = blob.download_as_string()
    download_time = time.time() - download_start
    logger.info(f"⏱️ 다운로드 완료: {download_time:.2f}초 ({file_size_kb/download_time:.2f} KB/초)")
    
    # JSON 파싱 시간 측정
    parse_start = time.time()
    raw_data = json.loads(raw_content)
    parse_time = time.time() - parse_start
    logger.info(f"📊 JSON 파싱: {parse_time:.2f}초, {len(raw_data)}개 항목 로드됨")
    
    # 메모리 사용량 확인
    after_load_memory = get_memory_usage()
    memory_diff = after_load_memory - initial_memory
    logger.debug(f"📊 로드 후 메모리: {after_load_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
    
    # 텍스트 전처리 시작
    process_start = time.time()
    logger.info(f"🔄 텍스트 전처리 시작 (총 {len(raw_data)}개 항목)...")
    
    texts = []
    empty_count = 0
    success_count = 0
    
    # 진행 상황 표시
    for i, item in enumerate(tqdm(raw_data, desc=f"전처리 진행")):
        # 1000개마다 로깅
        if i > 0 and i % 1000 == 0:
            progress = i / len(raw_data) * 100
            elapsed = time.time() - process_start
            items_per_sec = i / elapsed
            estimated_total = elapsed / i * len(raw_data)
            remaining = max(0, estimated_total - elapsed)
            
            logger.info(f"  → {i}/{len(raw_data)} 처리 중 ({progress:.1f}%) - "
                      f"속도: {items_per_sec:.1f}개/초, 남은 시간: {remaining:.1f}초")
        
        try:
            title = item.get("title") or ""
            selftext = item.get("selftext") or ""
            full_text = f"{title.strip()} {selftext.strip()}".strip()
            
            if full_text:
                texts.append(full_text)
                success_count += 1
            else:
                empty_count += 1
        except Exception as e:
            logger.warning(f"⚠️ 항목 {i} 처리 오류: {e}")
    
    process_time = time.time() - process_start
    total_time = time.time() - start_time
    
    # 처리 결과 통계
    logger.info(f"⏱️ 전처리 완료: {process_time:.2f}초 ({len(raw_data)/process_time:.1f}개/초)")
    logger.info(f"📈 처리 결과: 총 {len(raw_data)}개 중 {success_count}개 성공, {empty_count}개 빈 항목")
    logger.info(f"⏱️ 총 소요 시간: {total_time:.2f}초")
    
    # 텍스트 샘플 출력
    if texts:
        sample_text = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
        logger.debug(f"📝 샘플 텍스트: {sample_text}")
    
    return texts

def save_texts_to_gcs(bucket_name, gcs_path, texts):
    start_time = time.time()
    logger.info(f"💾 GCS 저장 시작: {len(texts)}개 텍스트")
    
    # JSON 변환 시간 측정
    json_start = time.time()
    json_content = json.dumps(texts, ensure_ascii=False, indent=2)
    json_time = time.time() - json_start
    
    content_size_kb = len(json_content) / 1024
    logger.info(f"⏱️ JSON 변환: {json_time:.2f}초, 크기: {content_size_kb:.2f} KB")
    
    # GCS 업로드 시간 측정
    upload_start = time.time()
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    blob.upload_from_string(
        data=json_content,
        content_type='application/json'
    )
    
    upload_time = time.time() - upload_start
    total_time = time.time() - start_time
    
    logger.info(f"⏱️ 업로드 완료: {upload_time:.2f}초 ({content_size_kb/upload_time:.1f} KB/초)")
    logger.info(f"⏱️ 총 저장 시간: {total_time:.2f}초")
    logger.info(f"📤 저장 완료: gs://{bucket_name}/{gcs_path}")
    
    return content_size_kb

if __name__ == "__main__":
    # 전체 시작 시간
    total_start_time = time.time()
    initial_memory = get_memory_usage()
    
    logger.info(f"🚀 전체 서브레딧 전처리 시작 - 날짜: {today}")
    logger.info(f"💻 시스템 정보: CPU {psutil.cpu_percent()}%, 메모리 {psutil.virtual_memory().percent}%")
    logger.info(f"📋 처리 대상: {len(SUBREDDITS)}개 서브레딧")
    
    # 실행 결과 통계용 변수
    results = {
        "success": 0,
        "not_found": 0,
        "error": 0,
        "total_items": 0,
        "total_texts": 0,
        "total_size_kb": 0
    }
    
    subreddit_stats = {}
    
    # 서브레딧 처리 진행율 표시
    for idx, sub in enumerate(SUBREDDITS):
        sub_start_time = time.time()
        
        # 진행율 표시
        progress = (idx / len(SUBREDDITS)) * 100
        elapsed = time.time() - total_start_time
        remaining = 0 if idx == 0 else (elapsed / idx) * (len(SUBREDDITS) - idx)
        
        logger.info("=" * 60)
        logger.info(f"🔄 진행률: {progress:.1f}% ({idx+1}/{len(SUBREDDITS)}) - "
                  f"예상 남은 시간: {remaining/60:.1f}분")
        logger.info(f"📂 처리 대상: r/{sub}")
        
        input_path = f"sns/reddit/{sub}/{today}/accumulated.json"
        output_path = f"sns/reddit/{sub}/{today}/reddit_text.json"
        
        logger.info(f"📥 입력: gs://{BUCKET_NAME}/{input_path}")
        logger.info(f"📤 출력: gs://{BUCKET_NAME}/{output_path}")
        logger.info("-" * 60)

        try:
            texts = preprocess_reddit_from_gcs(BUCKET_NAME, input_path)
            size_kb = save_texts_to_gcs(BUCKET_NAME, output_path, texts)
            
            sub_time = time.time() - sub_start_time
            logger.info(f"✅ r/{sub} 전처리 완료: {len(texts)}개 텍스트, 소요 시간: {sub_time:.2f}초")
            
            # 통계 업데이트
            results["success"] += 1
            results["total_texts"] += len(texts)
            results["total_size_kb"] += size_kb
            
            # 서브레딧별 통계
            subreddit_stats[sub] = {
                "status": "success",
                "texts": len(texts),
                "size_kb": size_kb,
                "time": sub_time
            }
            
        except FileNotFoundError:
            logger.warning(f"⚠️ r/{sub}: accumulated.json 없음 → 건너뜀")
            results["not_found"] += 1
            subreddit_stats[sub] = {"status": "not_found", "time": time.time() - sub_start_time}
            
        except Exception as e:
            logger.error(f"❌ r/{sub} 처리 중 오류 발생: {e}", exc_info=True)
            results["error"] += 1
            subreddit_stats[sub] = {"status": "error", "error": str(e), "time": time.time() - sub_start_time}
    
    # 총 소요 시간 계산
    total_time = time.time() - total_start_time
    final_memory = get_memory_usage()
    memory_diff = final_memory - initial_memory
    
    # 최종 결과 요약
    logger.info("=" * 60)
    logger.info(f"🎉 모든 서브레딧 전처리 완료!")
    logger.info(f"⏱️ 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
    logger.info(f"📊 메모리 사용량: {final_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
    logger.info(f"📈 처리 결과:")
    logger.info(f"  • 성공: {results['success']}/{len(SUBREDDITS)}개 서브레딧")
    logger.info(f"  • 파일 없음: {results['not_found']}개")
    logger.info(f"  • 오류: {results['error']}개")
    logger.info(f"  • 총 텍스트 수: {results['total_texts']}개")
    logger.info(f"  • 총 데이터 크기: {results['total_size_kb']:.2f} KB")
    
    # 서브레딧별 상세 결과
    logger.info("\n🔍 서브레딧별 처리 결과:")
    for sub, stats in subreddit_stats.items():
        status = "✅ 성공" if stats["status"] == "success" else "⚠️ 파일 없음" if stats["status"] == "not_found" else "❌ 오류"
        details = ""
        if stats["status"] == "success":
            details = f"{stats['texts']}개 텍스트, {stats['size_kb']:.2f} KB"
        elif stats["status"] == "error":
            details = f"오류: {stats['error']}"
        
        logger.info(f"  • r/{sub}: {status} - {details} (소요 시간: {stats['time']:.2f}초)")
    
    logger.info("=" * 60)

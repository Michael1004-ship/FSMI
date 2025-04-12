from datetime import datetime, timedelta
import os
import json
import pandas as pd
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage
import time
import logging
import argparse
import sys
import concurrent.futures
from tqdm import tqdm
import random

# ----------------------------
# 설정
# ----------------------------
BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
REDDIT_PREFIX = "sns/reddit/"
SLEEP_SECONDS = 0.1  # 병렬 처리하므로 지연 시간 감소
SAVE_FILENAME = "reddit_anxiety_index.csv"

# 병렬 처리 설정
MAX_WORKERS = 8  # 동시 처리할 최대 스레드 수

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
logger = logging.getLogger("reddit_finbert")

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------
logger.info("🧠 FinBERT 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
logger.info("✅ 모델 로딩 완료")

# ----------------------------
# GCS 연결
# ----------------------------
client = storage.Client()
bucket_raw = client.bucket(BUCKET_RAW)
bucket_index = client.bucket(BUCKET_INDEX)

# ----------------------------
# 서브레딧 목록
# ----------------------------
subreddits = [
    "dividends", "EconMonitor", "economics",
    "economy", "finance", "financialindependence", "investing", "MacroEconomics",
    "personalfinance", "StockMarket", "stocks", "wallstreetbets"
]

# ----------------------------
# 개별 텍스트 분석 함수 (병렬 처리용)
# ----------------------------
def process_text(item, subreddit):
    """단일 Reddit 게시물 분석 (병렬 처리용)"""
    try:
        text = item.get("title", "") + " " + item.get("selftext", "")
        text = text.strip()
        if not text:
            return {
                "success": False,
                "error": "텍스트 없음",
                "id": item.get("id", "unknown"),
                "subreddit": subreddit
            }
        
        # 감정 분석
        result = finbert(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        
        # 짧은 지연 (충돌 방지)
        time.sleep(SLEEP_SECONDS + random.uniform(0, 0.1))
        
        return {
            "success": True,
            "id": item.get("id", "unknown"),
            "subreddit": subreddit,
            "label": label,
            "score": score,
            "is_negative": label == "negative"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "id": item.get("id", "unknown"),
            "subreddit": subreddit
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

def process_subreddit(subreddit, items, executor):
    """서브레딧 데이터 병렬 처리"""
    logger.info(f"\n처리 중: r/{subreddit} ({len(items)}개 항목)")
    
    # 병렬 처리를 위한 작업 제출
    futures = [executor.submit(process_text, item, subreddit) for item in items]
    
    neg_scores = []
    failed_items = []
    
    # 진행 상황을 위한 tqdm
    with tqdm(total=len(futures), desc=f"r/{subreddit}", unit="텍스트") as pbar:
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            pbar.update(1)
            
            if result["success"]:
                if result["is_negative"]:
                    neg_scores.append(result["score"])
            else:
                failed_items.append({
                    "subreddit": result["subreddit"],
                    "id": result["id"],
                    "error": result["error"]
                })
    
    avg_score = sum(neg_scores) / len(neg_scores) if neg_scores else 0
    return {
        "subreddit": subreddit,
        "anxiety_score": avg_score,
        "negative_scores": neg_scores,
        "failed_items": failed_items
    }

# ----------------------------
# 메인 함수
# ----------------------------
def main(start_date, end_date, workers=MAX_WORKERS, debug=False):
    """메인 실행 함수"""
    # 날짜 범위로 된 JSON 파일 경로
    date_range_str = f"{start_date}_to_{end_date}"
    json_blob_path = f"{REDDIT_PREFIX}full/{date_range_str}.json"
    
    logger.info(f"📦 파일 검색 중: gs://{BUCKET_RAW}/{json_blob_path}")
    blob = bucket_raw.blob(json_blob_path)
    
    if not blob.exists():
        logger.error("🚫 해당 날짜의 Reddit 데이터 파일이 존재하지 않습니다.")
        return

    # 전체 시작 시간
    total_start_time = time.time()
    
    # 데이터 로드
    content = blob.download_as_bytes()
    all_data = json.load(BytesIO(content))
    logger.info(f"전체 {len(all_data)}개의 게시물 로드됨")
    
    # 날짜별로 데이터 분류
    date_data_map = {}
    for item in all_data:
        created_utc = datetime.fromtimestamp(item.get("created_utc", 0))
        date_str = created_utc.strftime("%Y-%m-%d")
        if date_str not in date_data_map:
            date_data_map[date_str] = []
        date_data_map[date_str].append(item)
    
    # 처리할 날짜 목록
    dates = list(date_data_map.keys())
    total_dates = len(dates)
    logger.info(f"총 {total_dates}개 날짜 처리 예정: {', '.join(dates)}")

    # 각 날짜별 처리
    for date_idx, target_date in enumerate(dates, 1):
        data = date_data_map[target_date]
        date_start_time = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"📅 {target_date} 처리 시작 ({len(data)}개 항목) - {date_idx}/{total_dates}")
        
        # 로그 디렉토리 설정
        target_log_dir = os.path.join(LOG_ROOT, target_date)
        os.makedirs(target_log_dir, exist_ok=True)

        results = []
        all_negative_scores = []
        all_failed_items = []

        # 서브레딧별로 데이터 분류
        subreddit_data = {}
        for item in data:
            subreddit = item.get("subreddit", "unknown")
            if subreddit not in subreddit_data:
                subreddit_data[subreddit] = []
            subreddit_data[subreddit].append(item)

        # ThreadPoolExecutor 설정
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # 각 서브레딧 처리
            for subreddit in subreddits:
                if subreddit not in subreddit_data:
                    logger.warning(f"[!] {subreddit} 데이터 없음")
                    continue

                subreddit_items = subreddit_data[subreddit]
                
                # 서브레딧 처리
                subreddit_result = process_subreddit(subreddit, subreddit_items, executor)
                results.append({
                    "subreddit": subreddit, 
                    "anxiety_score": subreddit_result["anxiety_score"]
                })
                all_negative_scores.extend(subreddit_result["negative_scores"])
                all_failed_items.extend(subreddit_result["failed_items"])

        # Anxiety Index 계산
        total_texts = len(data)  # 해당 날짜의 전체 텍스트 수
        negative_ratio = len(all_negative_scores) / total_texts if total_texts > 0 else 0
        average_negative_score = sum(all_negative_scores) / len(all_negative_scores) if all_negative_scores else 0
        anxiety_index = negative_ratio * average_negative_score

        # 결과 요약
        logger.info(f"\n📊 {target_date} 분석 결과:")
        logger.info(f"  • 총 게시물: {total_texts}개")
        logger.info(f"  • 부정 게시물: {len(all_negative_scores)}개")
        logger.info(f"  • 부정 비율: {negative_ratio:.3f}")
        logger.info(f"  • 평균 부정 점수: {average_negative_score:.3f}")
        logger.info(f"  • 불안 지수: {anxiety_index:.3f}")
        logger.info(f"  • 실패 수: {len(all_failed_items)}개")

        # 저장
        save_path = f"reddit/{target_date}/{SAVE_FILENAME}"
        temp_file = f"/tmp/reddit_anxiety_index_{target_date}.csv"

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("negative_ratio,average_negative_score,anxiety_index\n")
            f.write(f"{negative_ratio:.3f},{average_negative_score:.3f},{anxiety_index:.3f}\n\n")
            f.write("subreddit,anxiety_score\n")
            for row in results:
                f.write(f"{row['subreddit']},{row['anxiety_score']:.3f}\n")

        blob = bucket_index.blob(save_path)
        blob.upload_from_filename(temp_file)
        logger.info(f"✅ 저장 완료 → gs://{BUCKET_INDEX}/{save_path}")

        # 실패 로그 저장
        if all_failed_items:
            log_path = f"reddit/{target_date}/failed_items.json"
            temp_log_file = f"/tmp/failed_items_{target_date}.json"
            with open(temp_log_file, "w") as f:
                json.dump(all_failed_items, f, ensure_ascii=False, indent=2)
            log_blob = bucket_index.blob(log_path)
            log_blob.upload_from_filename(temp_log_file)
            logger.warning(f"⚠️ 실패 로그 저장됨 → gs://{BUCKET_INDEX}/{log_path}")
        
        # 날짜별 소요 시간 계산
        date_elapsed = time.time() - date_start_time
        logger.info(f"⏱️ {target_date} 처리 완료: {date_elapsed:.1f}초 ({date_elapsed/60:.1f}분)")
    
    # 전체 소요 시간 계산
    total_elapsed = time.time() - total_start_time
    logger.info(f"\n🎉 모든 날짜 처리 완료! 총 소요 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")

# ----------------------------
# CLI 인터페이스
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit FinBERT 감정 분석기")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"병렬 처리 스레드 수 (기본값: {MAX_WORKERS})")
    args = parser.parse_args()
    
    try:
        # 날짜 형식 검증
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
        if end_date < start_date:
            logger.error("❌ 종료일이 시작일보다 앞설 수 없습니다.")
            sys.exit(1)
            
        logger.info(f"📅 분석 기간: {args.start} ~ {args.end}")
        logger.info(f"🧵 병렬 처리 스레드 수: {args.workers}")
        
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.info("🔧 디버그 모드 활성화")
        
        main(args.start, args.end, args.workers, args.debug)
        
    except ValueError as e:
        logger.error(f"❌ 날짜 형식이 올바르지 않습니다: {str(e)}")
        sys.exit(1)
    except Exception as e:
        if args.debug:
            logger.exception("오류 발생:")
        else:
            logger.error(f"오류 발생: {str(e)}")
        sys.exit(1)

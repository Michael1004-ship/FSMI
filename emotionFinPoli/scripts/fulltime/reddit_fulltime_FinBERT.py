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

# ----------------------------
# 로깅 설정
# ----------------------------

# TARGET_DATE 설정 부분을 다음으로 교체
parser = argparse.ArgumentParser(description="Reddit FinBERT 분석기")
parser.add_argument("--date", required=True, help="분석 대상 날짜 (예: 2024-03-01)")
args = parser.parse_args()
TARGET_DATE = args.date

# 로그 디렉토리 설정 부분을 다음으로 교체
LOG_ROOT = "/home/hwangjeongmun691/logs"
LOG_DATE_DIR = os.path.join(LOG_ROOT, TARGET_DATE)
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/reddit_finbert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_finbert")

# ----------------------------
# 설정
# ----------------------------

BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
REDDIT_PREFIX = "sns/reddit/"
SLEEP_SECONDS = 1.0
SAVE_FILENAME = "reddit_anxiety_index.csv"

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------

logger.info("FinBERT 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
logger.info("모델 로딩 완료")

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

def main(start_date, end_date):
    # 날짜 범위로 된 JSON 파일 경로
    date_range_str = f"{start_date}_to_{end_date}"
    json_blob_path = f"{REDDIT_PREFIX}full/{date_range_str}.json"
    
    logger.info(f"📦 파일 검색 중: gs://{BUCKET_RAW}/{json_blob_path}")
    blob = bucket_raw.blob(json_blob_path)
    
    if not blob.exists():
        logger.error("🚫 해당 날짜의 Reddit 데이터 파일이 존재하지 않습니다.")
        return

    # 데이터 로드
    content = blob.download_as_bytes()
    all_data = json.load(BytesIO(content))
    
    # 날짜별로 데이터 분류
    date_data_map = {}
    for item in all_data:
        created_utc = datetime.fromtimestamp(item.get("created_utc", 0))
        date_str = created_utc.strftime("%Y-%m-%d")
        if date_str not in date_data_map:
            date_data_map[date_str] = []
        date_data_map[date_str].append(item)

    # 각 날짜별 처리
    for target_date, data in date_data_map.items():
        logger.info(f"\n=== {target_date} 처리 시작 ({len(data)}개 항목) ===")
        
        # 로그 디렉토리 설정
        LOG_DATE_DIR = os.path.join(LOG_ROOT, target_date)
        os.makedirs(LOG_DATE_DIR, exist_ok=True)

        results = []
        all_scores = []
        failed_items = []

        # 서브레딧별로 데이터 분류 및 처리
        subreddit_data = {}
        for item in data:
            subreddit = item.get("subreddit", "unknown")
            if subreddit not in subreddit_data:
                subreddit_data[subreddit] = []
            subreddit_data[subreddit].append(item)

        for subreddit in subreddits:
            if subreddit not in subreddit_data:
                logger.warning(f"[!] {subreddit} 데이터 없음")
                continue

            subreddit_items = subreddit_data[subreddit]
            logger.info(f"\n처리 중: r/{subreddit} ({len(subreddit_items)}개 항목)")

            neg_scores = []
            for idx, item in enumerate(subreddit_items, 1):
                try:
                    text = item.get("title", "") + " " + item.get("selftext", "")
                    text = text.strip()
                    if not text:
                        continue
                    
                    result = finbert(text[:512])[0]
                    label = result["label"].lower()
                    score = result["score"]
                    
                    if label == "negative":
                        neg_scores.append(score)
                        all_scores.append(score)
                    
                    time.sleep(SLEEP_SECONDS)
                    
                except Exception as e:
                    failed_items.append({
                        "subreddit": subreddit,
                        "id": item.get("id", "unknown"),
                        "error": str(e)
                    })
                    continue

            avg_score = sum(neg_scores) / len(neg_scores) if neg_scores else 0
            results.append({"subreddit": subreddit, "anxiety_score": avg_score})

        # Anxiety Index 계산
        total_texts = sum([len(json.load(BytesIO(bucket_raw.blob(f"{REDDIT_PREFIX}{sr}/{target_date}/accumulated.json").download_as_bytes())))
                        for sr in subreddits if bucket_raw.blob(f"{REDDIT_PREFIX}{sr}/{target_date}/accumulated.json").exists()])
        
        negative_ratio = len(all_scores) / total_texts if total_texts > 0 else 0
        average_negative_score = sum(all_scores) / len(all_scores) if all_scores else 0
        anxiety_index = negative_ratio * average_negative_score

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
        if failed_items:
            log_path = f"reddit/{target_date}/failed_items.json"
            temp_log_file = f"/tmp/failed_items_{target_date}.json"
            with open(temp_log_file, "w") as f:
                json.dump(failed_items, f, ensure_ascii=False, indent=2)
            log_blob = bucket_index.blob(log_path)
            log_blob.upload_from_filename(temp_log_file)
            logger.warning(f"⚠️ 실패 로그 저장됨 → gs://{BUCKET_INDEX}/{log_path}")

# ----------------------------
# CLI 인터페이스
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit FinBERT 감정 분석기")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    args = parser.parse_args()
    
    try:
        # 날짜 형식 검증
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
        if end_date < start_date:
            logger.error("❌ 종료일이 시작일보다 앞설 수 없습니다.")
            sys.exit(1)
            
        logger.info(f"📅 분석 기간: {args.start} ~ {args.end}")
        logger.info("🔧 FinBERT 모델 로딩 중...")
        
        main(args.start, args.end)
        
    except ValueError as e:
        logger.error(f"❌ 날짜 형식이 올바르지 않습니다: {str(e)}")
        sys.exit(1)
    except Exception as e:
        if args.debug:
            logger.exception("오류 발생:")
        else:
            logger.error(f"오류 발생: {str(e)}")
        sys.exit(1)

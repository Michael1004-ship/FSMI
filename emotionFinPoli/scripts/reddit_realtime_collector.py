from datetime import datetime, timedelta, time
# scripts/reddit_realtime_collector.py
import os
import json

import logging
import pandas as pd
import praw
from google.cloud import storage
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

# ✅ 설정
SUBREDDITS = [
    # 경제 관련
    "economics", "economy", "MacroEconomics", "EconMonitor",
    # 금융 관련
    "finance", "investing", "financialindependence", "personalfinance",
    "wallstreetbets", "stocks", "StockMarket", "dividends"
]

BUCKET_NAME = "emotion-raw-data"
STATE_FILE = "reddit_last_run.txt"
POST_LIMIT = 1000  # 각 서브레딧당 최대 수집 수 (기존 100 → 1000)

# ✅ 로깅 설정
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
        logging.FileHandler(f"{LOG_DATE_DIR}/reddit_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_collector")

# ✅ Reddit API 초기화 (.env 파일에서 로드)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ✅ 누적 데이터 로드
def load_existing_data(storage_client, subreddit, date_str):
    file_path = f"sns/reddit/{subreddit}/{date_str}/accumulated.json"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    if blob.exists():
        content = blob.download_as_string()
        return pd.DataFrame(json.loads(content))
    return pd.DataFrame(columns=['id', 'title', 'selftext', 'created_utc', 'score'])

# ✅ 저장
def save_to_gcs(df, subreddit, date_str):
    file_path = f"sns/reddit/{subreddit}/{date_str}/accumulated.json"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    logger.info(f"✅ 저장 완료: gs://{BUCKET_NAME}/{file_path}")

# ✅ 메인 실행

def run():
    # 현재 시간과 시간대 설정
    now = datetime.utcnow()
    hour = now.hour
    date_str = now.strftime("%Y-%m-%d")
    
    if hour < 13:
        # 오전 수집 → 전날 19:00 ~ 오늘 12:30
        start = datetime.combine((now - timedelta(days=1)).date(), time(hour=19))
        end = datetime.combine(now.date(), time(hour=12, minute=30))
    else:
        # 오후 수집 → 오늘 12:30 ~ 오늘 19:00
        start = datetime.combine(now.date(), time(hour=12, minute=30))
        end = datetime.combine(now.date(), time(hour=19))
    # ✅ 딱 이 두 구간만 매일 반복 수집되도록 설계됨
    # ✅ GDELT처럼 "now -1시간" 보정 없음
    # ✅ 시간 누락이나 중복 없이, 깔끔하게 "2회 수집 = 하루 전체 수집" 완성
    
    logger.info(f"Reddit 수집 시작: {date_str}, 시간 범위: {start} ~ {end}")

    for sub in SUBREDDITS:
        logger.info(f"📥 서브레딧: r/{sub}")
        posts = []

        for submission in reddit.subreddit(sub).new(limit=POST_LIMIT):
            # 🔁 수집 시간 필터
            created_time = datetime.utcfromtimestamp(submission.created_utc)
            if not (start <= created_time <= end):
                continue
                
            posts.append({
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "created_utc": created_time.isoformat(),
                "score": submission.score
            })

        new_df = pd.DataFrame(posts)
        if new_df.empty:
            logger.info(f"🔸 신규 글 없음: r/{sub}")
            continue

        storage_client = storage.Client()
        existing_df = load_existing_data(storage_client, sub, date_str)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['id'])
        save_to_gcs(combined_df, sub, date_str)

    logger.info("🎉 Reddit 수집 완료!")

if __name__ == "__main__":
    run()
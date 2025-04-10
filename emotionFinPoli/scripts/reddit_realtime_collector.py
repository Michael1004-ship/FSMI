# scripts/reddit_realtime_collector.py
import os
import json
import datetime
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
POST_LIMIT = 100  # 각 서브레딧당 최대 수집 수

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
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    logger.info(f"Reddit 수집 시작: {date_str}")

    for sub in SUBREDDITS:
        logger.info(f"📥 서브레딧: r/{sub}")
        posts = []

        for submission in reddit.subreddit(sub).new(limit=POST_LIMIT):
            posts.append({
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "created_utc": datetime.datetime.utcfromtimestamp(submission.created_utc).isoformat(),
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
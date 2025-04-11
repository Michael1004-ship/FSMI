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

# ----------------------------
# 설정
# ----------------------------
BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
MODEL_NAME = "yiyanghkust/finbert-tone"
LOCAL_TEMP_DIR = "/tmp"
GDELT_PREFIX = "news/gdelt/"

# ----------------------------
# 로그 설정
# ----------------------------
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/gdelt_finbert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_finbert")

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ----------------------------
# GCS 연결
# ----------------------------
client = storage.Client()
bucket_raw = client.bucket(BUCKET_RAW)
bucket_index = storage.Client().bucket(BUCKET_INDEX)

# ----------------------------
# 본문 크롤링 함수
# ----------------------------
def fetch_article_text(url):
    try:
        config = newspaper.Config()
        config.browser_user_agent = "Mozilla/5.0"
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        logger.warning(f"[X] 크롤링 실패: {url} | {e}")
        return None

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
def main(start_date, end_date, debug=False):
    """메인 실행 함수"""
    date_list = get_date_range(start_date, end_date)
    logger.info(f"처리할 날짜: {date_list}")

    for date_str in date_list:
        # 해당 날짜의 폴더에서 JSON 파일 찾기
        prefix_path = f"{GDELT_PREFIX}full/{date_str}_to_{date_str}.json"
        blob = bucket_raw.blob(prefix_path)
        
        if not blob.exists():
            logger.warning(f"🚫 {date_str} 날짜의 데이터가 없습니다.")
            continue

        try:
            content = blob.download_as_bytes()
            items = json.load(BytesIO(content))
            
            if not items:
                logger.warning(f"⚠️ {date_str}에 처리할 뉴스가 없습니다.")
                continue

            logger.info(f"📅 {date_str} - 총 {len(items)}개 기사 분석 시작")
            results = []
            details = []

            for item in items:
                if "DocumentIdentifier" in item:
                    url = item["DocumentIdentifier"]
                    text = fetch_article_text(url)
                    if not text:
                        continue
                    try:
                        result = finbert(text[:512])[0]
                        label = result["label"].lower()
                        score = result["score"]
                        details.append([url, label, score])
                        if label == "negative":
                            results.append(score)
                    except Exception as e:
                        logger.error(f"FinBERT 분석 실패: {e}")
                        continue

            total_count = len(details)
            negative_ratio = len(results) / total_count if total_count > 0 else 0
            avg_negative_score = sum(results) / len(results) if results else 0
            anxiety_index = negative_ratio * avg_negative_score

            # 저장
            save_folder = f"news/{date_str}"
            save_filename = "news_anxiety_index.csv"
            local_path = os.path.join(LOCAL_TEMP_DIR, f"{date_str}_{save_filename}")
            with open(local_path, "w", encoding="utf-8") as f:
                f.write("negative_ratio,average_negative_score,anxiety_index\n")
                f.write(f"{negative_ratio:.3f},{avg_negative_score:.3f},{anxiety_index:.3f}\n\n")
                f.write("url,label,score\n")
                for row in details:
                    f.write(f"{row[0]},{row[1]},{row[2]:.3f}\n")

            blob = bucket_index.blob(f"{save_folder}/{save_filename}")
            blob.upload_from_filename(local_path)
            logger.info(f"✅ 저장 완료 → gs://{BUCKET_INDEX}/{save_folder}/{save_filename}")

        except Exception as e:
            logger.error(f"❌ {date_str} 처리 중 오류 발생: {str(e)}")
            if debug:
                logger.exception("상세 오류:")
            continue

# ----------------------------
# CLI 인터페이스
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT 뉴스 감정 분석기")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    args = parser.parse_args()
    
    try:
        main(args.start, args.end, args.debug)
    except Exception as e:
        if args.debug:
            logger.exception("오류 발생:")
        else:
            logger.error(f"오류 발생: {str(e)}")
        sys.exit(1)

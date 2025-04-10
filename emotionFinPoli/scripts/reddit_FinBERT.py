from datetime import datetime
import os
import json
import pandas as pd
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage
import time
import logging

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
TARGET_DATE = datetime.utcnow().strftime("%Y-%m-%d")
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

# ----------------------------
# 서브레딧별 FinBERT 분석
# ----------------------------

results = []
all_scores = []
failed_items = []  # 실패 추적용 리스트

logger.info(f"분석 시작: {TARGET_DATE}")

try:
    for subreddit in subreddits:
        logger.info(f"\n처리 중: r/{subreddit}")
        
        try:
            blob_path = f"{REDDIT_PREFIX}{subreddit}/{TARGET_DATE}/accumulated.json"
            blob = bucket_raw.blob(blob_path)
            if not blob.exists():
                logger.warning(f"[!] {subreddit} 없음: {blob_path}")
                continue

            data = json.load(BytesIO(blob.download_as_bytes()))
            logger.info(f"데이터 로드됨: {len(data)}개 항목")

            neg_scores = []
            for idx, item in enumerate(data, 1):
                try:
                    logger.info(f"항목 처리 중 ({idx}/{len(data)})")
                    text = item.get("title", "") + " " + item.get("selftext", "")
                    text = text.strip()
                    if not text:
                        logger.warning(f"빈 텍스트 건너뜀: item {idx}")
                        continue
                    
                    logger.info(f"텍스트 길이: {len(text)} 글자")
                    result = finbert(text[:512])[0]
                    logger.info(f"분석 결과: {result['label']} ({result['score']:.3f})")
                    
                    if result["label"].lower() == "negative":  # 대소문자 구분 없이 비교
                        neg_scores.append(result["score"])
                        all_scores.append(result["score"])
                    
                    time.sleep(SLEEP_SECONDS)
                    
                except Exception as e:
                    logger.error(f"[!] 항목 분석 실패 (item {idx}): {str(e)}")
                    failed_items.append({
                        "subreddit": subreddit,
                        "id": item.get("id", "unknown"),
                        "error": str(e)
                    })
                    continue

            avg_score = sum(neg_scores) / len(neg_scores) if neg_scores else 0
            results.append({"subreddit": subreddit, "anxiety_score": avg_score})
            
            if neg_scores:
                logger.info(f"✓ {subreddit}: {len(neg_scores)}개 분석됨 (평균 점수: {avg_score:.3f})")
            else:
                logger.warning(f"! {subreddit}: 분석된 항목 없음")
                
        except Exception as e:
            logger.error(f"[!] 서브레딧 처리 실패 ({subreddit}): {str(e)}")
            continue

    # ----------------------------
    # Anxiety Index 계산
    # ----------------------------

    total_texts = sum([len(json.load(BytesIO(bucket_raw.blob(f"{REDDIT_PREFIX}{sr}/{TARGET_DATE}/accumulated.json").download_as_bytes())))
                       for sr in subreddits if bucket_raw.blob(f"{REDDIT_PREFIX}{sr}/{TARGET_DATE}/accumulated.json").exists()])
    negative_ratio = len(all_scores) / total_texts if total_texts > 0 else 0
    average_negative_score = sum(all_scores) / len(all_scores) if all_scores else 0
    anxiety_index = negative_ratio * average_negative_score

    # ----------------------------
    # CSV 저장: anxiety 요약 + 서브레딧별 상세
    # ----------------------------

    save_path = f"reddit/{TARGET_DATE}/{SAVE_FILENAME}"
    temp_file = "/tmp/reddit_anxiety_index.csv"

    with open(temp_file, "w", encoding="utf-8") as f:
        # 1부: 요약
        f.write("negative_ratio,average_negative_score,anxiety_index\n")
        f.write(f"{negative_ratio:.3f},{average_negative_score:.3f},{anxiety_index:.3f}\n\n")

        # 2부: 서브레딧별 anxiety score
        f.write("subreddit,anxiety_score\n")
        for row in results:
            f.write(f"{row['subreddit']},{row['anxiety_score']:.3f}\n")

    # GCS 업로드
    blob = bucket_index.blob(save_path)
    blob.upload_from_filename(temp_file)
    logger.info(f"✅ 저장 완료 → gs://{BUCKET_INDEX}/{save_path}")

    # 실패 로그 저장
    if failed_items:
        log_path = f"reddit/{TARGET_DATE}/failed_items.json"
        temp_log_file = "/tmp/failed_items.json"
        with open(temp_log_file, "w") as f:
            json.dump(failed_items, f, ensure_ascii=False, indent=2)
        log_blob = bucket_index.blob(log_path)
        log_blob.upload_from_filename(temp_log_file)
        logger.warning(f"⚠️ 실패 로그 저장됨 → gs://{BUCKET_INDEX}/{log_path}")

    # ----------------------------
    # 결과 요약
    # ----------------------------

    logger.info("\n=== 분석 결과 요약 ===")
    logger.info(f"총 처리된 서브레딧: {len(results)}개")
    logger.info(f"총 분석된 텍스트: {len(all_scores)}개")
    logger.info(f"전체 텍스트 수: {total_texts}개")
    logger.info(f"부정 비율: {negative_ratio:.3f}")
    logger.info(f"평균 부정 점수: {average_negative_score:.3f}")
    logger.info(f"불안 지수: {anxiety_index:.3f}")
    logger.info(f"실패한 항목: {len(failed_items)}개")

    # 주기적으로 임시 결과 저장 (예: 100개 항목마다)
    if len(all_scores) % 100 == 0 and len(all_scores) > 0:
        checkpoint_file = f"/tmp/reddit_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump({"results": results, "scores": all_scores}, f)
        logger.info(f"체크포인트 저장됨: {checkpoint_file}")
except Exception as e:
    logger.error(f"치명적 오류 발생: {e}")
    # 중간 결과 저장
    if results:
        emergency_file = f"/tmp/reddit_emergency_{int(time.time())}.json"
        with open(emergency_file, "w") as f:
            json.dump({"results": results, "scores": all_scores}, f)
        logger.error(f"응급 백업 저장됨: {emergency_file}")

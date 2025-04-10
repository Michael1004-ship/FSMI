import pandas as pd
import numpy as np
from google.cloud import storage
import argparse
from pathlib import Path
import datetime
from io import StringIO
import logging
import os

# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/build_anxiety_index.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("build_anxiety_index")


def compute_anxiety_score(df):
    """
    Compute the anxiety score based on the negative ratio and average negative score.
    Args:
        df: DataFrame containing the data.
    Returns:
        Computed anxiety score (negative_ratio * average_negative_score ^ 1.5).
    """
    ratio = float(df.iloc[1, 1])  # negative_ratio
    avg_score = float(df.iloc[1, 2])  # average_negative_score
    logger.info(f"Negative Ratio: {ratio}, Average Negative Score: {avg_score}")  # 디버깅용 출력
    anxiety_score = ratio * (avg_score ** 1.5)  # 비선형 처리
    return ratio, avg_score, anxiety_score


def compute_std(df, source):
    """
    Compute standard deviation for a given column based on source.
    Args:
        df: DataFrame containing the data.
        source: Data source ('news' or 'reddit_*').
    Returns:
        Standard deviation and scores for the column.
    """
    if source == "news":
        scores = df.iloc[4:, 2].astype(float)  # 뉴스는 3열 (index 2)
    else:
        scores = df.iloc[4:, 1].astype(float)  # 레딧은 2열 (index 1)
    return scores.std(), scores


def load_and_process(bucket_name, blob_path, source: str):
    """
    Load CSV file from GCS, compute the anxiety score, and Z-scores.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        raise FileNotFoundError(f"File not found in GCS: gs://{bucket_name}/{blob_path}")
    
    content = blob.download_as_text()
    df = pd.read_csv(StringIO(content), header=None)

    ratio, avg_score, anxiety_score = compute_anxiety_score(df)
    std, scores = compute_std(df, source)
    z_scores = (scores - avg_score) / std
    z_scores = np.clip(z_scores, -5, 5)  # ✅ 클리핑 적용: Z-score가 -5~5 사이로 제한
    z_scores.name = source + "_z"

    return ratio, avg_score, std, anxiety_score, z_scores


def upload_to_gcs(bucket_name, destination_blob_path, local_file_path):
    """
    Upload the final anxiety index CSV to GCS.
    Args:
        bucket_name: Name of the GCS bucket.
        destination_blob_path: Destination path in the GCS bucket.
        local_file_path: Local file path to upload.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    blob.upload_from_filename(local_file_path)
    logger.info(f"⛅ Uploaded to GCS: gs://{bucket_name}/{destination_blob_path}")


def check_if_already_processed(bucket_name, destination_blob_path):
    """
    최종 결과 파일이 이미 존재하는지 확인합니다.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)
    return blob.exists()


def build_anxiety_index(date_str):
    """
    Build the final anxiety index by processing news and reddit data.
    Args:
        date_str: The date string (YYYY-MM-DD) for the folder name.
    """
    bucket_name = "emotion-index-data"
    destination_blob_path = f"final_anxiety_index/{date_str}/anxiety_index_final.csv"
    
    # 이미 처리된 파일인지 확인하고 로그만 출력
    if check_if_already_processed(bucket_name, destination_blob_path):
        logger.warning(f"⚠️ 해당 날짜({date_str})의 지수가 이미 존재합니다. 파일을 덮어씁니다.")
        logger.info(f"기존 파일: gs://{bucket_name}/{destination_blob_path}")
    
    # 나머지 처리 계속 진행
    news_blob_path = f"news/{date_str}/news_anxiety_index.csv"
    reddit_finbert_blob_path = f"reddit/{date_str}/reddit_anxiety_index.csv"  # FinBERT
    reddit_roberta_blob_path = f"reddit/{date_str}/reddit_anxiety_roberta.csv"  # RoBERTa

    # 뉴스 데이터 처리
    news_ratio, news_avg, news_std, news_anx, news_z = load_and_process(bucket_name, news_blob_path, "news")

    # 레딧 데이터 처리 (FinBERT와 RoBERTa)
    rf_ratio, rf_avg, rf_std, rf_anx, rf_z = load_and_process(bucket_name, reddit_finbert_blob_path, "reddit_finbert")
    rr_ratio, rr_avg, rr_std, rr_anx, rr_z = load_and_process(bucket_name, reddit_roberta_blob_path, "reddit_roberta")

    # 레딧 데이터 결합 (FinBERT + RoBERTa)
    combined_reddit_z = np.exp((rf_z + rr_z) / 2)
    reddit_combined_score = combined_reddit_z.mean()
    
    # ✅ 여기서만 클리핑
    news_z_clipped = news_z.clip(-3, 3)
    reddit_z_clipped = combined_reddit_z.clip(-3, 3)
    
    # 최종 지수 계산
    final_index_series = np.exp(0.3 * news_z_clipped + 0.7 * reddit_z_clipped)
    final_index = final_index_series.replace([np.inf, -np.inf], np.nan).dropna().mean()

    # 로컬에 저장 후 GCS 업로드
    output_dir = Path("output") / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "anxiety_index_final.csv"
    
    # 기본 DataFrame 생성 (수정: 모든 행을 한 번에 정의)
    final_df = pd.DataFrame([
        ["Total", "", "", "", final_index],
        ["News", news_ratio, news_avg, news_std, news_anx],
        ["Reddit_FinBERT", rf_ratio, rf_avg, rf_std, rf_anx],
        ["Reddit_RoBERTa", rr_ratio, rr_avg, rr_std, rr_anx],
        ["Reddit_Combined", "", "", "", reddit_combined_score]
    ], columns=["Type", "Ratio", "Avg Score", "Std", "Anxiety Index"])
    
    final_df.to_csv(output_file, index=False)

    logger.info(f"✅ Saved locally to: {output_file}")

    # GCS에 업로드
    upload_to_gcs(bucket_name, destination_blob_path, str(output_file))


if __name__ == "__main__":
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # UTC 기준 사용
    
    parser = argparse.ArgumentParser(description="Build Anxiety Index from emotion scores")
    parser.add_argument("--date", default=today, help=f"Date folder in format YYYY-MM-DD (default: today {today})")
    args = parser.parse_args()
    
    logger.info(f"Building anxiety index for date: {args.date}")
    build_anxiety_index(args.date)

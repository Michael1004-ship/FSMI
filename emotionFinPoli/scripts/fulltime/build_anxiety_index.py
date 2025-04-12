from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from google.cloud import storage
import argparse
from pathlib import Path
from io import StringIO
import logging
import os
import sys
import concurrent.futures
from tqdm import tqdm
import time

# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 디렉토리 생성
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
logger = logging.getLogger("build_anxiety_index")

# 병렬 처리 설정
MAX_WORKERS = 4  # 동시 처리할 최대 날짜 수

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
    logger.debug(f"Negative Ratio: {ratio}, Average Negative Score: {avg_score}")  # 디버깅용 출력
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

    if source == "news":
        # ✅ 뉴스는 비중 * 강도 기반으로 anxiety_score 시계열 생성
        ratio = float(df.iloc[1, 1])
        avg_score = float(df.iloc[1, 2])
        logger.debug(f"[NEWS] Negative Ratio: {ratio}, Avg Score: {avg_score}")

        # 뉴스 시계열 점수는 4번째 행부터 시작 (부정 문장들의 감정 점수)
        scores = df.iloc[4:, 2].astype(float)
        # anxiety_score 시계열: 비중 * 점수^1.5
        anxiety_scores = ratio * (scores ** 1.5)
        mean_anx = anxiety_scores.mean()
        std_anx = anxiety_scores.std()
        z_scores = (anxiety_scores - mean_anx) / std_anx
        z_scores = np.clip(z_scores, -3, 3)  # ✅ 클리핑
        z_scores = np.exp(z_scores)         # ✅ 지수화
        z_scores.name = "news_z"

        return ratio, avg_score, std_anx, anxiety_scores.mean(), z_scores
    else:
        # Reddit 데이터 처리 (기존 방식 유지)
        ratio = float(df.iloc[1, 1])  # negative_ratio
        avg_score = float(df.iloc[1, 2])  # average_negative_score
        logger.debug(f"[REDDIT] {source} - Ratio: {ratio}, Avg Score: {avg_score}")
        
        scores = df.iloc[4:, 1].astype(float)  # 레딧은 2열 (index 1)
        std = scores.std()
        z_scores = (scores - avg_score) / std
        z_scores = np.clip(z_scores, -5, 5)  # ✅ 클리핑
        z_scores.name = source + "_z"
        
        anxiety_score = ratio * (avg_score ** 1.5)  # 비선형 처리
        
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
    Returns:
        Dict with status and result information
    """
    start_time = time.time()
    result_info = {
        "date": date_str,
        "success": False,
        "error": None,
        "index": None
    }
    
    try:
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

        logger.info(f"✅ {date_str} - 로컬 저장 완료: {output_file}")

        # GCS에 업로드
        upload_to_gcs(bucket_name, destination_blob_path, str(output_file))
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ {date_str} - 불안 지수: {final_index:.4f} (처리 시간: {elapsed_time:.1f}초)")
        
        result_info["success"] = True
        result_info["index"] = final_index
        result_info["elapsed_time"] = elapsed_time
        
        return result_info
        
    except FileNotFoundError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ {date_str} 처리 실패: 필요한 파일을 찾을 수 없습니다")
        logger.error(str(e))
        
        result_info["success"] = False
        result_info["error"] = f"FileNotFoundError: {str(e)}"
        result_info["elapsed_time"] = elapsed_time
        
        return result_info
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"❌ {date_str} 처리 중 오류 발생")
        logger.error(str(e))
        
        result_info["success"] = False
        result_info["error"] = str(e)
        result_info["elapsed_time"] = elapsed_time
        
        return result_info


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

def process_date_range(start_date, end_date, workers=MAX_WORKERS, debug=False):
    """주어진 날짜 범위의 데이터를 병렬 처리"""
    date_list = get_date_range(start_date, end_date)
    logger.info(f"처리할 날짜: {date_list} (총 {len(date_list)}일)")
    logger.info(f"병렬 처리 스레드 수: {workers}개")
    
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.info("🔧 디버그 모드 활성화")
    
    results = []
    success_count = 0
    failed_dates = []
    
    # 전체 시작 시간
    total_start_time = time.time()
    
    # 병렬 처리
    with tqdm(total=len(date_list), desc="불안 지수 계산 중", unit="일") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # 날짜별 작업 제출
            future_to_date = {executor.submit(build_anxiety_index, date): date for date in date_list}
            
            # 결과 처리
            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["success"]:
                        success_count += 1
                    else:
                        failed_dates.append(date)
                    
                except Exception as e:
                    logger.error(f"❌ {date} 처리 중 예외 발생: {str(e)}")
                    failed_dates.append(date)
                
                # 진행 상황 업데이트
                pbar.update(1)
                
                # 주기적 상태 보고
                if pbar.n % max(1, len(date_list) // 5) == 0 or pbar.n == len(date_list):
                    elapsed = time.time() - total_start_time
                    remaining = (elapsed / pbar.n) * (len(date_list) - pbar.n) if pbar.n > 0 else 0
                    
                    logger.info(f"🔄 진행률: {pbar.n/len(date_list)*100:.1f}% ({pbar.n}/{len(date_list)}) - "
                              f"성공: {success_count}, 실패: {len(failed_dates)} - "
                              f"예상 남은 시간: {remaining/60:.1f}분")
    
    # 전체 소요 시간 계산
    total_elapsed = time.time() - total_start_time
    
    # 최종 결과 요약
    logger.info("\n" + "="*50)
    logger.info("📊 불안 지수 계산 최종 결과")
    logger.info(f"• 처리 기간: {start_date} ~ {end_date}")
    logger.info(f"• 총 처리 날짜: {len(date_list)}일")
    logger.info(f"• 성공: {success_count}일")
    logger.info(f"• 실패: {len(failed_dates)}일")
    
    if failed_dates:
        logger.info("• 실패한 날짜:")
        for date in failed_dates:
            logger.info(f"  - {date}")
    
    # 성공한 결과에 대한 지수 통계
    successful_indices = [r["index"] for r in results if r["success"] and r["index"] is not None]
    if successful_indices:
        avg_index = sum(successful_indices) / len(successful_indices)
        min_index = min(successful_indices)
        max_index = max(successful_indices)
        
        logger.info(f"• 평균 불안 지수: {avg_index:.4f}")
        logger.info(f"• 최소 불안 지수: {min_index:.4f}")
        logger.info(f"• 최대 불안 지수: {max_index:.4f}")
    
    logger.info(f"• 총 소요 시간: {total_elapsed:.1f}초 ({total_elapsed/60:.1f}분)")
    logger.info("="*50)
    
    return success_count, len(failed_dates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="불안 지수 계산 도구 (병렬 처리)")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"병렬 처리 스레드 수 (기본값: {MAX_WORKERS})")
    args = parser.parse_args()
    
    try:
        process_date_range(args.start, args.end, args.workers, args.debug)
    except Exception as e:
        if args.debug:
            logger.exception("오류 발생:")
        else:
            logger.error(f"오류 발생: {str(e)}")
        sys.exit(1)

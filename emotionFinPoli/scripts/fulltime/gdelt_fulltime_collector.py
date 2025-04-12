from datetime import datetime, timedelta
# scripts/gdelt_full_collector.py
import os
import requests
import zipfile
import io
import pandas as pd

import logging
import json
from google.cloud import storage
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_fulltime_collector")

TOP_DOMAINS = [
    "nytimes.com", "wsj.com", "bloomberg.com", "cnbc.com", "cnn.com",
    "ft.com", "reuters.com", "finance.yahoo.com", "forbes.com", "marketwatch.com"
]

FINANCE_KEYWORDS = [
    "market", "stock", "stocks", "equity", "shares", "finance", "financing",
    "interest rate", "interest rates", "bond", "bonds", "treasury",
    "investment", "investments", "investor", "investors",
    "fed", "federal reserve", "monetary policy",
    "nasdaq", "dow jones", "s&p", "sp500", "volatility",
    "bank", "banking", "credit", "debt",
    "economic", "economy", "gdp", "growth", "recession", "inflation",
    "deflation", "stagflation", "employment", "unemployment", "jobless",
    "labor market", "consumer confidence", "cpi", "ppi", "retail sales",
    "housing market", "housing prices", "industrial production"
]

BUCKET_NAME = "emotion-raw-data"

# GKG 컬럼 정의
GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds",
    "Quotations", "AllNames", "Amounts", "TranslationInfo",
    "Extras", "OriginalXML"
]

# 전체 URL 리스트 생성
MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# 병렬 처리 관련 설정
MAX_WORKERS = 10  # 동시 처리할 최대 스레드 수

def get_urls_between(start_date, end_date):
    logger.info("🔎 masterfilelist.txt 로드 중...")
    res = requests.get(MASTER_URL)
    lines = res.text.strip().split('\n')
    urls = []
    for line in lines:
        if ".gkg.csv.zip" in line:
            parts = line.strip().split()
            url = parts[2]
            timestamp = url.split("/")[-1].split(".")[0]
            dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            if start_date <= dt <= end_date:
                urls.append(url)
    logger.info(f"총 {len(urls)}개의 GKG URL 수집 예정")
    return urls

def download_and_parse_csv(url):
    """단일 URL 다운로드 및 파싱 처리 (병렬 처리용)"""
    try:
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    chunk_size = 10000
                    chunks = [chunk for chunk in pd.read_csv(f, sep='\t', header=None, 
                                                          dtype=str, chunksize=chunk_size,
                                                          encoding='latin-1',  
                                                          on_bad_lines='skip')]
                    df = pd.concat(chunks, ignore_index=True)
                    
                    # 컬럼 이름 설정
                    if len(df.columns) == len(GKG_COLUMNS):
                        df.columns = GKG_COLUMNS
                    elif len(df.columns) == len(GKG_COLUMNS) - 1:
                        df.columns = GKG_COLUMNS[:-1]
                    else:
                        df.columns = [f"Column_{i}" for i in range(len(df.columns))]
                    
                    # 금융 뉴스 필터링
                    filtered_df = df[df['DocumentIdentifier'].str.contains('http', na=False)]
                    filtered_df = filtered_df[filtered_df['DocumentIdentifier'].str.contains('|'.join(TOP_DOMAINS), case=False, na=False)]
                    filtered_df = filtered_df[filtered_df['V2Themes'].str.contains('|'.join(FINANCE_KEYWORDS), case=False, na=False)]
                    filtered_df = filtered_df.drop_duplicates(subset=['DocumentIdentifier'])
                    
                    return {
                        "url": url,
                        "success": True,
                        "data": filtered_df,
                        "original_size": len(df),
                        "filtered_size": len(filtered_df)
                    }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e)
        }

def save_to_gcs(df, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    logger.info(f"✅ 저장 완료: gs://{BUCKET_NAME}/{file_name}")

def run(start_str, end_str, max_workers=MAX_WORKERS):
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")

    # URL 수집
    logger.info(f"🔎 {start_str}부터 {end_str}까지의 GDELT URL 수집 중...")
    urls = get_urls_between(start_dt, end_dt + timedelta(days=1))
    logger.info(f"📋 총 {len(urls)}개의 GDELT URL 수집 완료")
    
    if len(urls) == 0:
        logger.warning("❌ 수집할 URL이 없습니다. 날짜 범위를 확인하세요.")
        return
    
    # 병렬 처리를 통한 URL 다운로드 및 파싱
    logger.info(f"⏳ 병렬 처리 시작 (최대 {max_workers}개 스레드)")
    start_time = datetime.now()
    
    all_data = []
    failed_urls = []
    success_count = 0
    total_original_size = 0
    total_filtered_size = 0
    
    # 진행 상황 표시를 위한 tqdm 설정
    with tqdm(total=len(urls), desc="GDELT 파일 처리 중", unit="파일") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # URL별로 작업 제출
            future_to_url = {executor.submit(download_and_parse_csv, url): url for url in urls}
            
            # 결과 처리
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result["success"]:
                        success_count += 1
                        all_data.append(result["data"])
                        total_original_size += result["original_size"]
                        total_filtered_size += result["filtered_size"]
                        
                        # 성공 로그는 상세 레벨에서만 출력
                        logger.debug(f"✓ 파일 처리 완료: {url} - 원본: {result['original_size']}행, 필터링 후: {result['filtered_size']}행")
                    else:
                        failed_urls.append({"url": url, "error": result["error"]})
                        logger.warning(f"⚠️ 처리 실패: {url} - {result['error']}")
                except Exception as e:
                    failed_urls.append({"url": url, "error": str(e)})
                    logger.warning(f"⚠️ 처리 실패: {url} - {str(e)}")
                
                # 진행 상황 업데이트
                pbar.update(1)
                
                # 주기적 상태 보고 (10%마다)
                if pbar.n % max(1, len(urls) // 10) == 0 or pbar.n == len(urls):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    progress = pbar.n / len(urls) * 100
                    remaining = (elapsed / pbar.n) * (len(urls) - pbar.n) if pbar.n > 0 else 0
                    logger.info(f"🔄 진행률: {progress:.1f}% ({pbar.n}/{len(urls)}) - "
                               f"성공: {success_count}, 실패: {len(failed_urls)} - "
                               f"예상 남은 시간: {remaining/60:.1f}분")
    
    # 처리 결과 요약
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"⏱️ 처리 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    logger.info(f"📊 처리 결과: 총 {len(urls)}개 중 {success_count}개 성공, {len(failed_urls)}개 실패")
    
    if all_data:
        try:
            final_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"📊 최종 데이터셋: {len(final_df)}개 금융 뉴스 (중복 제거 전)")
            
            # 중복 제거
            final_df = final_df.drop_duplicates(subset=['DocumentIdentifier'])
            logger.info(f"📊 중복 제거 후: {len(final_df)}개 금융 뉴스")
            
            # 날짜별 데이터 분포 분석
            if 'DATE' in final_df.columns:
                df_copy = final_df.copy()
                df_copy['date_only'] = df_copy['DATE'].str[:8]  # YYYYMMDDHHMMSS 형식에서 YYYYMMDD 추출
                date_counts = df_copy['date_only'].value_counts().sort_index()
                logger.info("📅 날짜별 뉴스 분포:")
                for date, count in date_counts.items():
                    date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                    logger.info(f"  • {date_formatted}: {count}개")
            
            # GCS에 저장
            fname = f"news/gdelt/full/{start_str}_to_{end_str}.json"
            logger.info(f"💾 GCS에 저장 중: {fname}")
            save_to_gcs(final_df, fname)
            logger.info(f"🎉 모든 처리 완료 - {len(urls)}개 파일에서 {len(final_df)}개 뉴스 저장")
            
            # 필터링 효율성 보고
            if total_original_size > 0:
                filter_ratio = total_filtered_size / total_original_size * 100
                logger.info(f"🔍 필터링 효율: 원본 {total_original_size}행 → 필터링 후 {total_filtered_size}행 ({filter_ratio:.2f}%)")
        except Exception as e:
            logger.error(f"❌ 최종 데이터 처리 중 오류 발생: {str(e)}")
    else:
        logger.warning("⚠️ 저장할 뉴스가 없습니다. 필터링 조건을 확인하세요.")
    
    # 실패한 URL 로그
    if failed_urls:
        logger.warning(f"⚠️ {len(failed_urls)}개 URL 처리 실패")
        # 실패 목록을 파일로 저장
        fail_log_path = f"{LOG_DATE_DIR}/gdelt_failed_urls_{start_str}_to_{end_str}.json"
        with open(fail_log_path, 'w') as f:
            json.dump(failed_urls, f, indent=2)
        logger.info(f"📝 실패 목록 저장됨: {fail_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT 데이터 수집기 (병렬 처리)")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"병렬 처리 스레드 수 (기본값: {MAX_WORKERS})")
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("🔧 디버그 모드 활성화됨")
    
    logger.info(f"🧵 병렬 처리 스레드 수: {args.workers}")
    
    try:
        run(args.start, args.end, args.workers)
    except Exception as e:
        if args.debug:
            logger.exception("❌ 치명적 오류 발생:")
        else:
            logger.error(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)
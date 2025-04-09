# scripts/gdelt_full_collector.py
import os
import requests
import zipfile
import io
import pandas as pd
import datetime
import logging
import json
from google.cloud import storage
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gdelt_full_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_full_collector")

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
            dt = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            if start_date <= dt <= end_date:
                urls.append(url)
    logger.info(f"총 {len(urls)}개의 GKG URL 수집 예정")
    return urls

def download_and_parse_csv(zip_url):
    try:
        logger.info(f"다운로드: {zip_url}")
        res = requests.get(zip_url, timeout=60)
        res.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    chunk_size = 10000
                    chunks = [chunk for chunk in pd.read_csv(f, sep='\t', header=None, dtype=str, chunksize=chunk_size)]
                    return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        logger.warning(f"{zip_url} 실패: {e}")
        return None

def set_column_names(df):
    if len(df.columns) == len(GKG_COLUMNS):
        df.columns = GKG_COLUMNS
    elif len(df.columns) == len(GKG_COLUMNS) - 1:
        df.columns = GKG_COLUMNS[:-1]
    else:
        df.columns = [f"Column_{i}" for i in range(len(df.columns))]
    return df

def filter_finance_news(df):
    df = df[df['DocumentIdentifier'].str.contains('http', na=False)]
    df = df[df['DocumentIdentifier'].str.contains('|'.join(TOP_DOMAINS), case=False, na=False)]
    df = df[df['V2Themes'].str.contains('|'.join(FINANCE_KEYWORDS), case=False, na=False)]
    return df.drop_duplicates(subset=['DocumentIdentifier'])

def save_to_gcs(df, file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    logger.info(f"✅ 저장 완료: gs://{BUCKET_NAME}/{file_name}")

def run(start_str, end_str):
    start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d")

    # URL 수집 진행 상황 표시
    logger.info(f"🔎 {start_str}부터 {end_str}까지의 GDELT URL 수집 중...")
    urls = get_urls_between(start_dt, end_dt + datetime.timedelta(days=1))
    logger.info(f"📋 총 {len(urls)}개의 GDELT URL 수집 완료")
    
    # 빈 상태 확인
    if len(urls) == 0:
        logger.warning("❌ 수집할 URL이 없습니다. 날짜 범위를 확인하세요.")
        return
        
    all_dfs = []
    processed_count = 0
    start_time = datetime.datetime.now()
    
    # 진행률 표시
    logger.info(f"⏳ GDELT 데이터 다운로드 및 분석 시작 ({len(urls)}개 파일)")
    
    for i, url in enumerate(urls):
        # 주기적으로 진행 상황 보고
        processed_count = i + 1
        if processed_count % 10 == 0 or processed_count == 1 or processed_count == len(urls):
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            progress = processed_count / len(urls) * 100
            remaining = elapsed / processed_count * (len(urls) - processed_count) if processed_count > 0 else 0
            
            logger.info(f"🔄 진행률: {progress:.1f}% ({processed_count}/{len(urls)}) - " 
                        f"예상 남은 시간: {remaining/60:.1f}분")
        
        df = download_and_parse_csv(url)
        if df is not None:
            logger.info(f"✓ 파일 처리 완료: {url} - 행 수: {len(df)}")
            df = set_column_names(df)
            filtered_df = filter_finance_news(df)
            logger.info(f"  → 필터링 후 금융 뉴스: {len(filtered_df)}/{len(df)} ({len(filtered_df)/len(df)*100:.1f}%)")
            all_dfs.append(filtered_df)
            
    # 처리 결과 요약
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"📊 최종 데이터셋: {len(final_df)}개 금융 뉴스 (중복 제거 후)")
        
        # 날짜별 데이터 분포 분석
        if 'DATE' in final_df.columns:
            df_copy = final_df.copy()
            df_copy['date_only'] = df_copy['DATE'].str[:8]  # YYYYMMDDHHMMSS 형식에서 YYYYMMDD 추출
            date_counts = df_copy['date_only'].value_counts().sort_index()
            logger.info("📅 날짜별 뉴스 분포:")
            for date, count in date_counts.items():
                date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                logger.info(f"  • {date_formatted}: {count}개")
        
        fname = f"news/gdelt/full/{start_str}_to_{end_str}.json"
        logger.info(f"💾 GCS에 저장 중: {fname}")
        save_to_gcs(final_df, fname)
        logger.info(f"🎉 모든 처리 완료 - {len(urls)}개 파일에서 {len(final_df)}개 뉴스 저장")
    else:
        logger.warning("⚠️ 저장할 뉴스가 없습니다. 필터링 조건을 확인하세요.")

if __name__ == "__main__":
    # 명령줄 인자 확인
    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
    else:
        # 사용자 입력 받기
        print("GDELT 데이터 수집 도구")
        print("-" * 30)
        start = input("시작일 (YYYY-MM-DD 형식): ")
        end = input("종료일 (YYYY-MM-DD 형식): ")
    
    # 날짜 형식 검증 (YYYY-MM-DD)
    try:
        datetime.datetime.strptime(start, "%Y-%m-%d")
        datetime.datetime.strptime(end, "%Y-%m-%d")
    except ValueError:
        print("오류: 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")
        sys.exit(1)
    
    logger.info(f"수집 기간: {start} ~ {end}")
    run(start, end)
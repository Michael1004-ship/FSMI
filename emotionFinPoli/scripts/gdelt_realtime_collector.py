# scripts/gdelt_realtime_collector.py
import os
import requests
import zipfile
import io
import pandas as pd
import datetime
import logging
import time
import json
from google.cloud import storage

# 로깅 설정
import os
from datetime import datetime

# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/gdelt_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_collector")

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

# ✅ 수집된 시간 기록용 파일
STATE_FILE = "last_collected.txt"

# ✅ 15분 단위 시간 생성기
def generate_time_stamps(start_time, end_time):
    stamps = []
    current = start_time
    while current <= end_time:
        stamps.append(current.strftime("%Y%m%d%H%M%S"))
        current += datetime.timedelta(minutes=15)
    return stamps

# ✅ GDELT URL 생성
def build_gkg_url(timestamp):
    return f"http://data.gdeltproject.org/gdeltv2/{timestamp}.gkg.csv.zip"

# ✅ zip 파일 다운로드 및 파싱
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

# ✅ 컬럼 이름 설정
GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds",
    "Quotations", "AllNames", "Amounts", "TranslationInfo",
    "Extras", "OriginalXML"
]

def set_column_names(df):
    if len(df.columns) == len(GKG_COLUMNS):
        df.columns = GKG_COLUMNS
    elif len(df.columns) == len(GKG_COLUMNS) - 1:
        df.columns = GKG_COLUMNS[:-1]
    else:
        df.columns = [f"Column_{i}" for i in range(len(df.columns))]
    return df

# ✅ 필터링

def filter_finance_news(df):
    df = df[df['DocumentIdentifier'].str.contains('http', na=False)]
    df = df[df['DocumentIdentifier'].str.contains('|'.join(TOP_DOMAINS), case=False, na=False)]
    df = df[df['V2Themes'].str.contains('|'.join(FINANCE_KEYWORDS), case=False, na=False)]
    return df.drop_duplicates(subset=['DocumentIdentifier'])

# ✅ GCS 누적 저장

def load_existing_data(storage_client, date_str):
    file_name = f"news/gdelt/{date_str}/accumulated.json"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    if blob.exists():
        content = blob.download_as_string()
        return pd.DataFrame(json.loads(content))
    return pd.DataFrame(columns=['DATE', 'DocumentIdentifier', 'SourceCommonName', 'V2Themes', 'Tone'])

def save_accumulated_to_gcs(df, date_str):
    file_name = f"news/gdelt/{date_str}/accumulated.json"
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    logger.info(f"✅ 누적 저장 완료: gs://{BUCKET_NAME}/{file_name}")

# ✅ 메인 실행

def run():
    now = datetime.datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    yesterday_str = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    # 마지막 수집 시간 불러오기
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            last_ts = datetime.datetime.strptime(f.read().strip(), "%Y%m%d%H%M%S")
    else:
        last_ts = now.replace(hour=0, minute=0, second=0, microsecond=0)

    logger.info(f"🔁 수집 범위: {last_ts} ~ {now}")

    # 시간대 리스트 생성
    stamps = generate_time_stamps(last_ts, now)

    all_dfs = []
    for stamp in stamps:
        url = build_gkg_url(stamp)
        df = download_and_parse_csv(url)
        if df is not None:
            df = set_column_names(df)
            df = filter_finance_news(df)
            all_dfs.append(df)

    storage_client = storage.Client()
    
    if all_dfs:
        new_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"📰 신규 뉴스 수집 수: {len(new_df)}")

        existing_df = load_existing_data(storage_client, date_str)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['DocumentIdentifier'])
        save_accumulated_to_gcs(combined_df, date_str)

        # 수집 시간 업데이트
        with open(STATE_FILE, 'w') as f:
            f.write(now.strftime("%Y%m%d%H%M%S"))

    else:
        logger.info("새로운 뉴스가 없습니다. 최근 데이터를 사용합니다.")
        
        # 오늘 데이터 확인
        today_data_exists = storage_client.bucket(BUCKET_NAME).blob(f"news/gdelt/{date_str}/accumulated.json").exists()
        
        if not today_data_exists:
            # 어제 데이터 확인
            yesterday_blob = storage_client.bucket(BUCKET_NAME).blob(f"news/gdelt/{yesterday_str}/accumulated.json")
            
            if yesterday_blob.exists():
                logger.info(f"어제({yesterday_str}) 데이터를 오늘({date_str}) 폴더에 복사합니다.")
                content = yesterday_blob.download_as_string()
                
                # 오늘 폴더에 저장
                today_blob = storage_client.bucket(BUCKET_NAME).blob(f"news/gdelt/{date_str}/accumulated.json")
                today_blob.upload_from_string(content, content_type='application/json')
                logger.info(f"✅ 데이터 복사 완료: gs://{BUCKET_NAME}/news/gdelt/{date_str}/accumulated.json")
            else:
                # 최근 3일 내 데이터 찾기
                found_data = False
                for i in range(2, 7):
                    check_date = (now - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                    check_blob = storage_client.bucket(BUCKET_NAME).blob(f"news/gdelt/{check_date}/accumulated.json")
                    
                    if check_blob.exists():
                        logger.info(f"{check_date} 데이터를 오늘({date_str}) 폴더에 복사합니다.")
                        content = check_blob.download_as_string()
                        today_blob = storage_client.bucket(BUCKET_NAME).blob(f"news/gdelt/{date_str}/accumulated.json")
                        today_blob.upload_from_string(content, content_type='application/json')
                        logger.info(f"✅ 데이터 복사 완료: gs://{BUCKET_NAME}/news/gdelt/{date_str}/accumulated.json")
                        found_data = True
                        break
                
                if not found_data:
                    # 최근 데이터가 없으면 빈 JSON 파일 생성
                    logger.warning("최근 데이터를 찾을 수 없어 빈 파일을 생성합니다.")
                    empty_df = pd.DataFrame(columns=['DATE', 'DocumentIdentifier', 'SourceCommonName', 'V2Themes', 'Tone'])
                    save_accumulated_to_gcs(empty_df, date_str)

if __name__ == "__main__":
    run()

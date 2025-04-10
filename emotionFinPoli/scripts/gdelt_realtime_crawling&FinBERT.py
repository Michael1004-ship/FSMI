import os
import re
import time
import json
import pandas as pd
from io import BytesIO
from datetime import datetime
import sys
import subprocess
import logging

from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from google.cloud import storage
import newspaper

# ----------------------------
# 설정
# ----------------------------

BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"
GDELT_PREFIX = "news/gdelt/"
SLEEP_SECONDS = 2.5
TARGET_DATE = datetime.utcnow().strftime("%Y-%m-%d")  # UTC 기준 사용
SAVE_FILENAME = "news_anxiety_index.csv"

# 실패한 URL 리스트 수집용
failed_urls = []

# 경로 설정
PROJECT_ROOT = "/home/hwangjeongmun691/projects/emotionFinPoli"
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
BUILD_INDEX_DIR = os.path.join(PROJECT_ROOT, "building_index")

# ----------------------------
# 모델 로딩 (FinBERT)
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ----------------------------
# GCS 연결
# ----------------------------

client = storage.Client()
bucket_raw = client.bucket(BUCKET_RAW)
bucket_index = client.bucket(BUCKET_INDEX)

# ----------------------------
# 날짜 폴더 내 JSON 파일 찾기
# ----------------------------

prefix_path = f"{GDELT_PREFIX}{TARGET_DATE}/"
blobs = client.list_blobs(BUCKET_RAW, prefix=prefix_path)

urls = []
for blob in blobs:
    if not blob.name.endswith(".json"):
        continue
    content = blob.download_as_bytes()
    try:
        items = json.load(BytesIO(content))
        for item in items:
            if "DocumentIdentifier" in item:
                urls.append(item["DocumentIdentifier"])
    except Exception as e:
        print(f"[!] JSON 파싱 오류: {blob.name} - {e}")
        continue

print(f"🔗 총 {len(urls)}개의 기사 URL 수집됨")

# ----------------------------
# 본문 크롤링 및 분석
# ----------------------------

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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/gdelt_finbert.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_finbert")

def fetch_article_text(url):
    try:
        print(f"[*] 크롤링 시도 중: {url}")  # 진행 상황 표시
        config = newspaper.Config()
        config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        
        article = Article(url, config=config)
        print(f"[*] 다운로드 시작...")
        article.download()
        print(f"[*] 파싱 시작...")
        article.parse()
        text = article.text.strip()
        print(f"[+] 성공: {len(text)} 글자 추출")
        return text
    except Exception as e:
        print(f"[X] 크롤링 실패: {url}")
        print(f"[X] 에러 타입: {type(e).__name__}")
        print(f"[X] 에러 내용: {str(e)}")
        failed_urls.append(url)  # 실패한 URL 추가
        return None

# 본문 크롤링 부분도 수정
print("\n크롤링 시작...")
results = []   # 평균 계산용 (negative만)
details = []   # 전체 감정 분석 결과 저장용

for i, url in enumerate(urls, 1):
    print(f"\n[{i}/{len(urls)}] URL 처리 중...")
    text = fetch_article_text(url)
    time.sleep(SLEEP_SECONDS)

    if text:
        try:
            result = finbert(text[:512])[0]
            print(f"[+] 감성 분석 결과: {result['label']} ({result['score']:.3f})")
            
            # 평균 계산용: negative만 따로
            if result["label"] == "negative":
                results.append(result["score"])
            
            # 전체 기사 저장용: 무조건 저장
            details.append([url, result["label"], result["score"]])
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            failed_urls.append(url)  # FinBERT 분석 실패한 URL도 추가
            continue

# ----------------------------
# 결과 집계 및 저장
# ----------------------------

# 결과 저장 경로
save_path = f"news/{TARGET_DATE}/{SAVE_FILENAME}"
temp_file = "/tmp/news_anxiety_index.csv"

# 지수 계산 - 대소문자 구분 없이 비교
total_count = len(details)
negatives = [row for row in details if row[1].lower() == "negative"]  # 소문자로 변환하여 비교
negative_ratio = len(negatives) / total_count if total_count > 0 else 0
average_negative_score = sum(row[2] for row in negatives) / len(negatives) if negatives else 0
anxiety_index = negative_ratio * average_negative_score

# CSV 작성
with open(temp_file, "w", encoding="utf-8") as f:
    # 1부: 지수 요약
    f.write("negative_ratio,average_negative_score,anxiety_index\n")
    f.write(f"{negative_ratio:.3f},{average_negative_score:.3f},{anxiety_index:.3f}\n\n")

    # 2부: 감정 분석 결과
    f.write("url,label,score\n")
    for row in details:
        f.write(f"{row[0]},{row[1]},{row[2]:.3f}\n")

# 업로드
blob = bucket_index.blob(save_path)
blob.upload_from_filename(temp_file)

print(f"✅ 저장 완료 → gs://{BUCKET_INDEX}/{save_path}")

# 실패 로그 저장
if failed_urls:
    log_path = f"news/{TARGET_DATE}/failed_urls.txt"
    temp_log_file = "/tmp/failed_urls.txt"
    with open(temp_log_file, "w") as f:
        for url in failed_urls:
            f.write(url + "\n")
    log_blob = bucket_index.blob(log_path)
    log_blob.upload_from_filename(temp_log_file)
    print(f"⚠️ 실패한 URL 로그 저장됨 → gs://{BUCKET_INDEX}/{log_path}")
    print(f"총 {len(failed_urls)}개의 URL 처리 실패")

def run_script(script_path, description, args=None, retry=1):
    """스크립트를 실행하고 상세 디버깅 정보를 로깅합니다."""
    for attempt in range(1, retry+1):
        try:
            # 특수문자 경로 문제 해결을 위해 문자열로 전달
            script_path = str(script_path)
            
            logger.info(f"✨ {description} 시작 (시도 {attempt}/{retry})...")
            logger.debug(f"실행 명령: {sys.executable} \"{script_path}\" {args if args else ''}")
            
            # 특수문자가 있는 경로는 list 형태로 전달해 subprocess에서 자동 이스케이프 처리
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            # 경로 디버깅
            logger.debug(f"스크립트 파일 존재 여부: {os.path.exists(script_path)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # 나머지 코드...
        except Exception as e:
            logger.error(f"오류가 발생했습니다: {e}")
            failed_urls.append(url)  # FinBERT 분석 실패한 URL도 추가
            continue

def main():
    # ...
    
    # 절대 경로로 스크립트 실행
    gdelt_collector = os.path.join(SCRIPTS_DIR, "gdelt_realtime_collector.py")
    reddit_collector = os.path.join(SCRIPTS_DIR, "reddit_realtime_collector.py")
    gdelt_finbert = os.path.join(SCRIPTS_DIR, "gdelt_realtime_crawling&FinBERT.py")
    reddit_finbert = os.path.join(SCRIPTS_DIR, "reddit_FinBERT.py")
    reddit_roberta = os.path.join(SCRIPTS_DIR, "reddit_RoBERTa.py")
    build_index = os.path.join(BUILD_INDEX_DIR, "build_anxiety_index.py")
    
    # 스크립트 실행 전 존재 확인
    for script in [gdelt_collector, reddit_collector, gdelt_finbert, reddit_finbert, reddit_roberta, build_index]:
        if os.path.exists(script):
            logger.debug(f"✓ 스크립트 확인: {script}")
        else:
            logger.error(f"✗ 스크립트 없음: {script}")
    
    # 스크립트 실행
    gdelt_success = run_script(gdelt_collector, "GDELT 데이터 수집", retry=2)
    # 나머지 스크립트 실행...

from datetime import datetime
import json
import subprocess
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import logging
from openai import OpenAI

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
        logging.FileHandler(f"{LOG_DATE_DIR}/gpt_labeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gpt_labeling")

load_dotenv()  # .env 파일 자동 로드

# ✅ 오늘 날짜
DATE = datetime.utcnow().strftime("%Y-%m-%d")

# ✅ 소스 종류
SOURCES = ["news", "reddit"]

# ✅ 공통 GCS 설정
GCS_BUCKET = "emotion-index-data"

# ✅ GPT 감정 라벨링 함수
def label_cluster(cluster_id, sentences):
    prompt = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    messages = [
        {"role": "system", "content": "Read the following sentences and provide ONE single emotion label in English. Respond with only one word. Examples: anxiety, anticipation, anger, cynicism, etc."},
        {"role": "user", "content": prompt}
    ]
    
    # 재시도 설정
    max_retries = 5
    retry_delay = 2  # 초기 지연 시간 (초)
    
    for attempt in range(max_retries):
        try:
            # API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error(f"❌ API 키가 설정되지 않았습니다.")
                return "API 키 오류"
            
            # SDK 사용
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3
            )
            
            # SDK 응답 객체는 ChatCompletion 형식입니다
            label = response.choices[0].message.content.strip()
            logger.info(f"[cluster {cluster_id}] → {label}")
            return label
                
        except Exception as e:
            logger.error(f"❌ Error labeling cluster {cluster_id}: {e}")
            
            # 지수 백오프로 대기 후 재시도
            wait_time = retry_delay * (2 ** attempt)
            logger.info(f"⏳ {wait_time}초 후 재시도 ({attempt+1}/{max_retries})...")
            time.sleep(wait_time)
            continue
            
    return "Rate Limit"  # 모든 재시도 실패

# ✅ 단일 소스 처리 함수
def process_source(source):
    logger.info(f"\n📦 Processing: {source}")
    base_gcs_path = f"gs://{GCS_BUCKET}/{source}/{DATE}/"

    gcs_input = base_gcs_path + "cluster_representative_texts.json"
    gcs_output = base_gcs_path + "cluster_labels.json"

    local_input = Path(f"/tmp/{source}_cluster_representative_texts.json")
    local_output = Path(f"/tmp/{source}_cluster_labels.json")

    try:
        subprocess.run(["gsutil", "cp", gcs_input, str(local_input)], check=True)
        logger.info(f"✅ 다운로드 완료: {gcs_input}")
    except subprocess.CalledProcessError:
        logger.error(f"❌ 파일 없음 또는 다운로드 실패: {gcs_input}")
        return

    with open(local_input, "r", encoding="utf-8") as f:
        cluster_texts = json.load(f)

    cluster_labels = {}
    # 클러스터 ID별로 그룹화 (cluster_0, cluster_1 등)
    clusters = {}
    for key, value in cluster_texts.items():
        if "_" in key:
            parts = key.split("_")
            cluster_id = parts[1]  # cluster_0_top1 -> 0
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            # size는 숫자라서 제외
            if "size" not in key and isinstance(value, str) and value.strip():
                clusters[cluster_id].append(value)
    
    # 각 클러스터별로 GPT 라벨링
    for cluster_id, texts in clusters.items():
        if texts:  # 빈 리스트가 아닌 경우만 처리
            label = label_cluster(cluster_id, texts[:5])  # 최대 5개 텍스트만 사용
            cluster_labels[f"cluster_{cluster_id}"] = label
            time.sleep(3)  # API 요청 사이에 3초 지연 추가

    with open(local_output, "w", encoding="utf-8") as f:
        json.dump(cluster_labels, f, ensure_ascii=False, indent=2)

    subprocess.run(["gsutil", "cp", str(local_output), gcs_output], check=True)
    logger.info(f"✅ GCS 업로드 완료: {gcs_output}")

# ✅ 메인 실행
def main():
    for source in SOURCES:
        process_source(source)

if __name__ == "__main__":
    main()

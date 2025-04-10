import os
import json
import time
import datetime
import logging
import psutil
import multiprocessing
from io import BytesIO
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import storage
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed

# ----------------------------
# Configuration
# ----------------------------
BUCKET_NAME = "emotion-raw-data"
OUTPUT_BUCKET = "emotion-index-data"
SUBREDDITS = [
    "anxiety", "depression", "dividends", "EconMonitor",
    "economics", "economy", "finance", "financialindependence",
    "investing", "MacroEconomics", "offmychest", "personalfinance",
    "StockMarket", "stocks", "wallstreetbets"
]
DATE = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # 이 줄을 주석 처리
# DATE = "2025-04-08"  # 고정된 날짜로 설정
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
CPU_CORES = max(1, multiprocessing.cpu_count() - 1)  # 1개 코어는 남겨둠

# ----------------------------
# Logging Setup
# ----------------------------
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
        logging.FileHandler(f"{LOG_DATE_DIR}/reddit_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_clustering")

# ----------------------------
# Performance Monitoring
# ----------------------------
def get_memory_usage() -> float:
    """현재 프로세스의 메모리 사용량을 MB 단위로 반환"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB 단위로 변환

# ----------------------------
# GCS에서 데이터 불러오기
# ----------------------------
def load_subreddit_texts(bucket_name: str, subreddit: str, date: str) -> List[str]:
    """단일 서브레딧의 텍스트 데이터를 로드"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"sns/reddit/{subreddit}/{date}/reddit_text.json"
    blob = bucket.blob(blob_path)
    
    if blob.exists():
        logger.debug(f"📥 다운로드 중: {blob_path}")
        content = blob.download_as_bytes()
        try:
            return json.load(BytesIO(content))
        except Exception as e:
            logger.warning(f"⚠️ JSON 파싱 실패: {blob_path} - {e}")
            return []
    else:
        logger.warning(f"❌ 파일 없음: {blob_path}")
        return []

def load_all_reddit_texts(bucket_name: str, subreddits: List[str], date: str) -> Tuple[List[str], Dict[str, int]]:
    """병렬로 모든 서브레딧의 텍스트 데이터를 로드"""
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"🔄 Reddit 텍스트 데이터 로드 시작 (서브레딧 {len(subreddits)}개)")
    
    # 병렬 처리로 각 서브레딧 데이터 로드
    logger.info(f"💻 병렬 처리 시작 (코어 {CPU_CORES}개 사용)")
    
    results = Parallel(n_jobs=CPU_CORES)(
        delayed(load_subreddit_texts)(bucket_name, sub, date) 
        for sub in tqdm(subreddits, desc="서브레딧 로드")
    )
    
    # 결과 집계
    all_texts = []
    subreddit_info = []
    subreddit_counts = {}
    
    for i, texts in enumerate(results):
        subreddit = subreddits[i]
        count = len(texts)
        subreddit_counts[subreddit] = count
        
        if count > 0:
            all_texts.extend(texts)
            subreddit_info.extend([subreddit] * count)
            
        logger.info(f"  • r/{subreddit}: {count}개 문장")
    
    # 성능 측정
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_diff = final_memory - initial_memory
    
    logger.info(f"✅ 총 {len(all_texts)}개 문장 로드 완료")
    logger.info(f"⏱️ 로드 시간: {total_time:.2f}초")
    logger.info(f"📊 메모리 사용: {final_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
    
    return all_texts, subreddit_info, subreddit_counts

# ----------------------------
# 문장 임베딩
# ----------------------------
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """문장을 벡터로 변환"""
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"🧠 문장 임베딩 시작 (모델: {model_name})")
    
    model = SentenceTransformer(model_name)
    logger.info(f"✅ 모델 로드 완료: {model_name}")
    
    # 배치 크기 최적화
    batch_size = 32
    logger.info(f"⚙️ 배치 크기: {batch_size}, 총 {len(texts)}개 문장")
    
    # 임베딩 계산
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=batch_size,
        device='cuda' if hasattr(model, 'device') and model.device.type == 'cuda' else 'cpu'
    )
    
    # 성능 측정
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_diff = final_memory - initial_memory
    
    logger.info(f"✅ 임베딩 완료: 벡터 크기 {embeddings.shape}")
    logger.info(f"⏱️ 임베딩 시간: {total_time:.2f}초 ({len(texts)/total_time:.1f}문장/초)")
    logger.info(f"📊 메모리 사용: {final_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
    
    return embeddings

# ----------------------------
# Find Optimal K
# ----------------------------
def find_best_k(embeddings: np.ndarray, k_range=range(3, 11)) -> int:
    """실루엣 점수를 기반으로 최적의 클러스터 수 결정"""
    start_time = time.time()
    logger.info(f"🔍 최적 클러스터 수 탐색 시작 (범위: {k_range.start}-{k_range.stop-1})")
    
    scores = []
    best_score = -1
    best_k = k_range.start
    
    for k in tqdm(k_range, desc="클러스터 수 테스트"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        logger.info(f"  • k={k}, silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    total_time = time.time() - start_time
    logger.info(f"🏆 최적 클러스터 수: k={best_k} (score={best_score:.4f})")
    logger.info(f"⏱️ 탐색 시간: {total_time:.2f}초")
    
    return best_k

# ----------------------------
# Extract Representative Texts (TF-IDF Based)
# ----------------------------
def extract_representative_texts(texts: List[str], labels: List[int], k: int, top_n: int = 5) -> dict:
    """TF-IDF 점수 기반으로 대표적인 텍스트 선택"""
    start_time = time.time()
    logger.info(f"🔍 클러스터 대표 문장 선택 중 (TF-IDF 기반)...")
    
    result = {}
    
    for i in range(k):
        # 해당 클러스터에 속한 텍스트 가져오기
        cluster_indices = [j for j in range(len(texts)) if labels[j] == i]
        cluster_texts = [texts[j] for j in cluster_indices]
        
        if not cluster_texts:
            result[f"cluster_{i}_representative"] = ""
            result[f"cluster_{i}_size"] = 0
            logger.warning(f"  • 클러스터 {i}가 비어 있습니다")
            continue
        
        # 클러스터 크기 저장
        result[f"cluster_{i}_size"] = len(cluster_texts)
        logger.info(f"  • 클러스터 {i} 처리 중: {len(cluster_texts)}개 텍스트")
        
        try:
            # TF-IDF 계산
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            
            # 문서별 TF-IDF 점수 합계 (높을수록 더 대표적)
            scores = tfidf_matrix.sum(axis=1).A1
            
            # 상위 N개 문서 선택
            top_indices = scores.argsort()[::-1][:top_n]
            
            # 대표 텍스트 저장
            result[f"cluster_{i}_representative"] = cluster_texts[top_indices[0]]
            
            # 상위 N개 텍스트도 저장
            for j, idx in enumerate(top_indices):
                result[f"cluster_{i}_top{j+1}"] = cluster_texts[idx]
                
            logger.info(f"  • 클러스터 {i}에서 {top_n}개 대표 텍스트 선택 완료")
            
        except Exception as e:
            logger.error(f"클러스터 {i} 처리 중 오류 발생: {e}")
            result[f"cluster_{i}_representative"] = ""
            if len(cluster_texts) > 0:
                result[f"cluster_{i}_representative"] = cluster_texts[0]
    
    total_time = time.time() - start_time
    logger.info(f"✅ 대표 문장 선택 완료 (소요 시간: {total_time:.2f}초)")
    
    return result

# ----------------------------
# GCS 저장 함수
# ----------------------------
def save_to_gcs(bucket_name, gcs_path, content, content_type="text/csv"):
    """데이터를 GCS에 저장"""
    start_time = time.time()
    logger.info(f"💾 GCS 저장 시작: gs://{bucket_name}/{gcs_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(content, content_type=content_type)
    
    total_time = time.time() - start_time
    logger.info(f"✅ GCS 저장 완료: gs://{bucket_name}/{gcs_path} (시간: {total_time:.2f}초)")

# ----------------------------
# 임베딩 저장 함수
# ----------------------------
def save_embeddings(embeddings: np.ndarray, output_bucket: str, date: str):
    """임베딩 결과를 NumPy 형식으로 GCS에 저장"""
    start_time = time.time()
    logger.info(f"💾 임베딩 저장 시작 (NumPy 형식)...")

    # NumPy 형식으로 저장 (.npy)
    np_buffer = BytesIO()
    np.save(np_buffer, embeddings)
    np_buffer.seek(0)
    
    gcs_npy_path = f"reddit/{date}/embeddings.npy"
    save_to_gcs(output_bucket, gcs_npy_path, np_buffer.getvalue(), content_type="application/octet-stream")
    
    total_time = time.time() - start_time
    logger.info(f"✅ 임베딩 저장 완료: gs://{output_bucket}/{gcs_npy_path} (시간: {total_time:.2f}초)")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # 전체 시작 시간 및 메모리 사용량
    total_start_time = time.time()
    initial_memory = get_memory_usage()
    
    logger.info("=" * 60)
    logger.info("🚀 Reddit 클러스터링 시작")
    logger.info(f"📅 대상 날짜: {DATE}")
    logger.info(f"💻 시스템 정보: CPU {psutil.cpu_percent()}%, 메모리 {psutil.virtual_memory().percent}%")
    logger.info("=" * 60)
    
    try:
        # 1. 모든 서브레딧 텍스트 로드
        texts, subreddit_info, subreddit_counts = load_all_reddit_texts(BUCKET_NAME, SUBREDDITS, DATE)
        
        if not texts:
            logger.error("❌ 처리할 텍스트가 없습니다. 종료합니다.")
            exit(1)
        
        # 2. 문장 임베딩
        embeddings = embed_texts(texts, MODEL_NAME)
        
        # 임베딩 저장 (NumPy 형식만)
        save_embeddings(embeddings, OUTPUT_BUCKET, DATE)

        # 3. 최적 클러스터 수 찾기
        best_k = find_best_k(embeddings)
        
        # 4. 클러스터링 수행
        logger.info(f"🔄 KMeans 클러스터링 시작 (k={best_k})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별 분포 계산
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  • 클러스터 {u}: {c}개 문장 ({c/len(labels)*100:.1f}%)")
        
        # 5. 대표 문장 추출 (TF-IDF 기반)
        reps = extract_representative_texts(texts, labels, best_k)
        
        # 6. 클러스터링 결과 저장
        cluster_data = []
        for i in range(len(texts)):
            cluster_data.append({
                "text": texts[i],
                "cluster": int(labels[i]),
                "subreddit": subreddit_info[i] if i < len(subreddit_info) else ""
            })
        
        # JSON으로 저장
        cluster_data_json = json.dumps(cluster_data, ensure_ascii=False)
        save_to_gcs(OUTPUT_BUCKET, f"reddit/{DATE}/cluster_data.json", cluster_data_json, content_type="application/json")
        
        # 대표 문장 저장
        json_path = f"reddit/{DATE}/cluster_representative_texts.json"
        json_content = json.dumps(reps, indent=2, ensure_ascii=False)
        save_to_gcs(OUTPUT_BUCKET, json_path, json_content, content_type="application/json")
        
        # 클러스터 통계 요약
        cluster_stats = {"clusters": {}}
        for i in range(best_k):
            cluster_size = len([label for label in labels if label == i])
            
            # 이 클러스터에 속한 서브레딧별 분포 계산
            subreddit_dist = {}
            if len(subreddit_info) == len(texts):
                for idx, label in enumerate(labels):
                    if label == i:
                        sub = subreddit_info[idx]
                        subreddit_dist[sub] = subreddit_dist.get(sub, 0) + 1
            
            cluster_stats["clusters"][str(i)] = {
                "size": cluster_size,
                "percentage": cluster_size / len(labels) * 100,
                "representative": reps.get(f"cluster_{i}_representative", "")[:100] if isinstance(reps.get(f"cluster_{i}_representative"), str) else "",
                "subreddit_distribution": subreddit_dist
            }
        
        stats_path = f"reddit/{DATE}/cluster_stats.json"
        stats_content = json.dumps(cluster_stats, indent=2)
        save_to_gcs(OUTPUT_BUCKET, stats_path, stats_content, content_type="application/json")
        
        # 완료 통계
        total_time = time.time() - total_start_time
        final_memory = get_memory_usage()
        memory_diff = final_memory - initial_memory
        
        logger.info("=" * 60)
        logger.info("🎉 Reddit 클러스터링 전체 완료!")
        logger.info(f"⏱️ 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        logger.info(f"📊 메모리 사용량: {final_memory:.2f} MB (증가: {memory_diff:.2f} MB)")
        logger.info(f"📊 처리 결과:")
        logger.info(f"  • 총 처리 문장: {len(texts)}개")
        logger.info(f"  • 클러스터 수: {best_k}개")
        logger.info(f"  • 저장된 파일:")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/reddit/{DATE}/cluster_data.json")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/reddit/{DATE}/cluster_representative_texts.json")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/reddit/{DATE}/cluster_stats.json")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}", exc_info=True)

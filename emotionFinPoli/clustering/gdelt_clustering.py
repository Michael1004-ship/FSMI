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
DATE = datetime.datetime.utcnow().strftime("%Y-%m-%d")  # 이 줄을 주석 처리
# DATE = "2025-04-08"  # 고정된 날짜로 설정
MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
CPU_CORES = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free

# ----------------------------
# Logging Setup
# ----------------------------
import os
from datetime import datetime

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
        logging.FileHandler(f"{LOG_DATE_DIR}/gdelt_clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gdelt_clustering")

# ----------------------------
# Performance Monitoring
# ----------------------------
def get_memory_usage() -> float:
    """Returns current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

# ----------------------------
# Load Data from GCS
# ----------------------------
def load_news_texts(bucket_name: str, date: str) -> List[str]:
    """Load news texts from GCS"""
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"🔄 Starting to load news data for date: {date}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"news/gdelt/{date}/news_text.json"
    blob = bucket.blob(blob_path)
    
    if blob.exists():
        logger.debug(f"📥 Downloading: {blob_path}")
        content = blob.download_as_bytes()
        try:
            texts = json.load(BytesIO(content))
            
            # Performance metrics
            total_time = time.time() - start_time
            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            logger.info(f"✅ Successfully loaded {len(texts)} news sentences")
            logger.info(f"⏱️ Loading time: {total_time:.2f} seconds")
            logger.info(f"📊 Memory usage: {final_memory:.2f} MB (increase: {memory_diff:.2f} MB)")
            
            return texts
        except Exception as e:
            logger.error(f"❌ JSON parsing failed: {blob_path} - {e}")
            return []
    else:
        logger.error(f"❌ File not found: {blob_path}")
        return []

# ----------------------------
# Text Embedding
# ----------------------------
def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    """Converts sentences to vectors"""
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"🧠 Starting text embedding (model: {model_name})")
    
    model = SentenceTransformer(model_name)
    logger.info(f"✅ Model loaded: {model_name}")
    
    # Batch size optimization
    batch_size = 32
    logger.info(f"⚙️ Batch size: {batch_size}, total sentences: {len(texts)}")
    
    # Calculate embeddings
    embeddings = model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=batch_size,
        device='cuda' if hasattr(model, 'device') and model.device.type == 'cuda' else 'cpu'  # Use GPU if available
    )
    
    # Performance metrics
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_diff = final_memory - initial_memory
    
    logger.info(f"✅ Embedding completed: vector shape {embeddings.shape}")
    logger.info(f"⏱️ Embedding time: {total_time:.2f} seconds ({len(texts)/total_time:.1f} sentences/sec)")
    logger.info(f"📊 Memory usage: {final_memory:.2f} MB (increase: {memory_diff:.2f} MB)")
    
    return embeddings

# ----------------------------
# Save Embeddings
# ----------------------------
def save_embeddings(embeddings: np.ndarray, output_bucket: str, date: str):
    """Save embeddings in NumPy format to GCS"""
    start_time = time.time()
    logger.info(f"💾 Starting embeddings save (NumPy format)...")

    # Save in NumPy format (.npy)
    np_buffer = BytesIO()
    np.save(np_buffer, embeddings)
    np_buffer.seek(0)
    
    gcs_npy_path = f"news/{date}/embeddings.npy"
    save_to_gcs(output_bucket, gcs_npy_path, np_buffer.getvalue(), content_type="application/octet-stream")
    
    total_time = time.time() - start_time
    logger.info(f"✅ Embeddings save completed: gs://{output_bucket}/{gcs_npy_path} (time: {total_time:.2f} seconds)")

# ----------------------------
# Find Optimal K
# ----------------------------
def find_best_k(embeddings: np.ndarray, k_range=range(3, 11)) -> int:
    """Find optimal cluster count based on silhouette score"""
    start_time = time.time()
    logger.info(f"🔍 Starting optimal cluster search (range: {k_range.start}-{k_range.stop-1})")
    
    scores = []
    best_score = -1
    best_k = k_range.start
    
    for k in tqdm(k_range, desc="Testing cluster counts"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append(score)
        logger.info(f"  • k={k}, silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    total_time = time.time() - start_time
    logger.info(f"🏆 Optimal cluster count: k={best_k} (score={best_score:.4f})")
    logger.info(f"⏱️ Search time: {total_time:.2f} seconds")
    
    return best_k

# ----------------------------
# Extract Representative Texts (TF-IDF Based)
# ----------------------------
def extract_representative_texts(texts: List[str], labels: List[int], k: int, top_n: int = 5) -> dict:
    """Select representative texts based on TF-IDF scores"""
    start_time = time.time()
    logger.info(f"🔍 Selecting cluster representative texts using TF-IDF...")
    
    result = {}
    
    for i in range(k):
        # Get texts for this cluster
        cluster_indices = [j for j in range(len(texts)) if labels[j] == i]
        cluster_texts = [texts[j] for j in cluster_indices]
        
        if not cluster_texts:
            result[f"cluster_{i}_representative"] = ""
            result[f"cluster_{i}_size"] = 0
            logger.warning(f"  • Cluster {i} is empty")
            continue
        
        # Store cluster size
        result[f"cluster_{i}_size"] = len(cluster_texts)
        logger.info(f"  • Processing cluster {i}: {len(cluster_texts)} texts")
        
        try:
            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            
            # Sum TF-IDF scores for each document (higher is more representative)
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Get top N documents by TF-IDF score
            top_indices = scores.argsort()[::-1][:top_n]
            
            # Store representative texts
            result[f"cluster_{i}_representative"] = cluster_texts[top_indices[0]]
            
            # Also store top N texts
            for j, idx in enumerate(top_indices):
                result[f"cluster_{i}_top{j+1}"] = cluster_texts[idx]
                
            logger.info(f"  • Selected {top_n} representative texts for cluster {i}")
            
        except Exception as e:
            logger.error(f"Error processing cluster {i}: {e}")
            result[f"cluster_{i}_representative"] = ""
            if len(cluster_texts) > 0:
                result[f"cluster_{i}_representative"] = cluster_texts[0]
    
    total_time = time.time() - start_time
    logger.info(f"✅ Representative text selection completed (time: {total_time:.2f} seconds)")
    
    return result

# ----------------------------
# Save to GCS
# ----------------------------
def save_to_gcs(bucket_name, gcs_path, content, content_type="text/csv"):
    """Save data to Google Cloud Storage"""
    start_time = time.time()
    logger.info(f"💾 Starting GCS save: gs://{bucket_name}/{gcs_path}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(content, content_type=content_type)
    
    total_time = time.time() - start_time
    logger.info(f"✅ GCS save completed: gs://{bucket_name}/{gcs_path} (time: {total_time:.2f} seconds)")

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Overall start time and memory usage
    total_start_time = time.time()
    initial_memory = get_memory_usage()
    
    logger.info("=" * 60)
    logger.info("🚀 Starting News Text Clustering")
    logger.info(f"📅 Target date: {DATE}")
    logger.info(f"💻 System info: CPU {psutil.cpu_percent()}%, Memory {psutil.virtual_memory().percent}%")
    logger.info("=" * 60)
    
    try:
        # 1. Load news texts
        texts = load_news_texts(BUCKET_NAME, DATE)
        
        if not texts:
            logger.error("❌ No texts to process. Exiting.")
            exit(1)
        
        # 2. Embed texts
        embeddings = embed_texts(texts, MODEL_NAME)

        # Save embeddings (NumPy format only)
        save_embeddings(embeddings, OUTPUT_BUCKET, DATE)
        
        # 3. Find optimal cluster count
        best_k = find_best_k(embeddings)
        
        # 4. Perform clustering
        logger.info(f"🔄 Starting KMeans clustering (k={best_k})")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  • Cluster {u}: {c} sentences ({c/len(labels)*100:.1f}%)")
        
        # 5. Extract representative texts using TF-IDF
        reps = extract_representative_texts(texts, labels, best_k)
        
        # 6. Save clustering results
        cluster_data = []
        for i in range(len(texts)):
            cluster_data.append({
                "text": texts[i],
                "cluster": int(labels[i])
            })
        
        # Save as JSON
        cluster_data_json = json.dumps(cluster_data, ensure_ascii=False)
        save_to_gcs(OUTPUT_BUCKET, f"news/{DATE}/cluster_data.json", cluster_data_json, content_type="application/json")
        
        # Save representative texts
        json_path = f"news/{DATE}/cluster_representative_texts.json"
        json_content = json.dumps(reps, indent=2, ensure_ascii=False)
        save_to_gcs(OUTPUT_BUCKET, json_path, json_content, content_type="application/json")
        
        # Cluster summary statistics
        cluster_stats = {"clusters": {}}
        for i in range(best_k):
            cluster_size = len([label for label in labels if label == i])
            cluster_stats["clusters"][str(i)] = {
                "size": cluster_size,
                "percentage": cluster_size / len(labels) * 100,
                "representative": reps.get(f"cluster_{i}_representative", "")[:100] if isinstance(reps.get(f"cluster_{i}_representative"), str) else ""
            }
        
        stats_path = f"news/{DATE}/cluster_stats.json"
        stats_content = json.dumps(cluster_stats, indent=2)
        save_to_gcs(OUTPUT_BUCKET, stats_path, stats_content, content_type="application/json")
        
        # Completion statistics
        total_time = time.time() - total_start_time
        final_memory = get_memory_usage()
        memory_diff = final_memory - initial_memory
        
        logger.info("=" * 60)
        logger.info("🎉 News clustering completed!")
        logger.info(f"⏱️ Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"📊 Memory usage: {final_memory:.2f} MB (increase: {memory_diff:.2f} MB)")
        logger.info(f"📊 Processing results:")
        logger.info(f"  • Total sentences processed: {len(texts)}")
        logger.info(f"  • Number of clusters: {best_k}")
        logger.info(f"  • Files saved:")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/news/{DATE}/cluster_data.json")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/news/{DATE}/cluster_representative_texts.json")
        logger.info(f"    - gs://{OUTPUT_BUCKET}/news/{DATE}/cluster_stats.json")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}", exc_info=True)

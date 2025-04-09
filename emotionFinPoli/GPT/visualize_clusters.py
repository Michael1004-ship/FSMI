import numpy as np
import matplotlib.pyplot as plt
import json
import umap
import os
from pathlib import Path
from datetime import datetime
import subprocess
from collections import Counter

# ✅ 오늘 날짜 (UTC 기준)
# date_str = datetime.utcnow().strftime("%Y-%m-%d")
date_str = "2025-04-08" # 고정 날짜 필요 시 사용

# ✅ 처리할 소스 목록
sources = ["news", "reddit"]

# ✅ 파일 이름 고정
files_needed = ["embeddings.npy", "cluster_data.json", "cluster_labels.json"]

# ✅ 공통 경로 포맷
base_gcs = "gs://emotion-index-data/{source}/{date}/{filename}"
local_tmp = "/tmp/{source}_{filename}"

def download_files(source):
    for fname in files_needed:
        gcs_path = base_gcs.format(source=source, date=date_str, filename=fname)
        local_path = local_tmp.format(source=source, filename=fname)
        try:
            subprocess.run(["gsutil", "cp", gcs_path, local_path], check=True)
            print(f"✅ 다운로드 완료: {gcs_path}")
        except subprocess.CalledProcessError:
            print(f"❌ 다운로드 실패: {gcs_path}")

def load_data(source):
    emb_path = local_tmp.format(source=source, filename="embeddings.npy")
    cluster_data_path = local_tmp.format(source=source, filename="cluster_data.json")
    cluster_labels_path = local_tmp.format(source=source, filename="cluster_labels.json")

    embeddings = np.load(emb_path)
    print(f"✅ 임베딩 로드 완료: {embeddings.shape}")

    with open(cluster_data_path, "r") as f:
        cluster_data = json.load(f)
    print(f"✅ 클러스터 데이터 로드 완료: {len(cluster_data)} 항목")
    # 디버깅: 첫 번째 항목 형식 확인
    if cluster_data:
        print(f"👉 첫 번째 항목 형식: {type(cluster_data[0])}")
        print(f"👉 첫 번째 항목 키: {cluster_data[0].keys() if isinstance(cluster_data[0], dict) else '딕셔너리 아님'}")

    with open(cluster_labels_path, "r") as f:
        cluster_labels = json.load(f)
    print(f"✅ 클러스터 라벨 로드 완료: {len(cluster_labels)} 항목")
    # 디버깅: 라벨 형식 확인
    print(f"👉 라벨 형식: {type(cluster_labels)}")
    
    # 🔧 자료형 보정
    cluster_labels = {str(k): v for k, v in cluster_labels.items()}

    return embeddings, cluster_data, cluster_labels

def visualize_umap(source, embeddings, cluster_data, cluster_labels):
    print(f"🔍 UMAP 시각화 시작 ({source})")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    print(f"✅ UMAP 차원 축소 완료: {embedding_2d.shape}")

    # 문장 순서에 맞춰 cluster_id 추출 (수정된 부분)
    # 기존 코드:
    # cluster_ids = [str(cluster_data[str(i)]) for i in range(len(embedding_2d))]
    
    # 수정된 코드:
    cluster_ids = [f"cluster_{item['cluster']}" for item in cluster_data]
    print(f"✅ 클러스터 ID 추출 완료: {len(cluster_ids)} 아이템")
    # 디버깅: 첫 5개 클러스터 ID 출력
    print(f"👉 처음 5개 클러스터 ID: {cluster_ids[:5]}")
    
    # 라벨 정보 출력
    print(f"👉 라벨 키 목록: {list(cluster_labels.keys())[:5]} 등...")
    
    cluster_names = [cluster_labels.get(cid, "Unknown") for cid in cluster_ids]
    print(f"✅ 클러스터 이름 매핑 완료")
    # 디버깅: 첫 5개 이름 출력
    print(f"👉 처음 5개 클러스터 이름: {cluster_names[:5]}")

    # 고유 라벨 색상 부여
    unique_labels = list(set(cluster_names))
    print(f"✅ 고유 라벨 {len(unique_labels)}개 발견: {unique_labels}")
    
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    point_colors = [label_to_color[label] for label in cluster_names]

    # 시각화
    print(f"🎨 시각화 플롯 생성 중...")
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, l in enumerate(cluster_names) if l == label]
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1], 
                    c=[label_to_color[label]], label=label, alpha=0.7, s=10)

    plt.title(f"UMAP Visualization of {source.capitalize()} Clusters ({date_str})")
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()

    # 저장 및 업로드
    local_img = f"/tmp/umap_{source}.png"
    gcs_img = f"gs://emotion-index-data/{source}/{date_str}/umap_plot.png"
    plt.savefig(local_img, dpi=300)
    print(f"✅ 이미지 저장 완료: {local_img}")
    
    subprocess.run(["gsutil", "cp", local_img, gcs_img], check=True)
    print(f"📤 업로드 완료: {gcs_img}")
    plt.close()

    # 📊 감정 비중 계산
    emotion_counts = Counter(cluster_names)
    emotions = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    total = sum(values)
    percentages = [round(v / total * 100, 1) for v in values]

    # ✅ .json 저장용 dict 생성
    emotion_ratio_dict = {emotion: {"count": count, "percentage": perc}
                          for emotion, count, perc in zip(emotions, values, percentages)}

    # 📁 로컬 + GCS 경로
    local_json = f"/tmp/emotion_ratio_{source}.json"
    gcs_json = f"gs://emotion-index-data/{source}/{date_str}/emotion_ratio.json"

    # 저장
    with open(local_json, "w") as f:
        json.dump(emotion_ratio_dict, f, indent=2)
    subprocess.run(["gsutil", "cp", local_json, gcs_json], check=True)
    print(f"✅ 감정 비중 JSON 저장 및 업로드 완료: {gcs_json}")

    # 🎨 시각화
    plt.figure(figsize=(8, 6))
    bars = plt.barh(emotions, percentages, color='skyblue')

    # 바 끝에 퍼센트 값 표시
    for bar, perc in zip(bars, percentages):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{perc}%", va='center', fontsize=9)

    plt.xlabel("Emotion Ratio (%)")
    plt.title(f"Emotion Distribution in {source.capitalize()} ({date_str})")
    plt.tight_layout()

    # 저장
    local_bar = f"/tmp/emotion_distribution_{source}.png"
    gcs_bar = f"gs://emotion-index-data/{source}/{date_str}/emotion_distribution.png"
    plt.savefig(local_bar, dpi=300)
    subprocess.run(["gsutil", "cp", local_bar, gcs_bar], check=True)
    print(f"📤 감정 비중 시각화 업로드 완료: {gcs_bar}")
    plt.close()

# ✅ 전체 실행
for src in sources:
    print(f"\n{'='*50}")
    print(f"🚀 {src.upper()} 데이터 처리 시작")
    print(f"{'='*50}")
    download_files(src)
    emb, cdata, clabels = load_data(src)
    visualize_umap(src, emb, cdata, clabels)
    print(f"{'='*50}")
    print(f"✅ {src.upper()} 처리 완료")
    print(f"{'='*50}\n")

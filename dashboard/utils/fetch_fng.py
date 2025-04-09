import requests
from datetime import datetime
import pandas as pd
from google.cloud import storage
import io

# 설정
BUCKET_NAME = 'emotion-index-data'
GCS_FOLDER_PREFIX = 'VIX_FNG'

def fetch_fng_score():
    try:
        url = 'https://api.alternative.me/fng/?limit=1&format=json'
        response = requests.get(url, timeout=10)
        data = response.json()
        score = int(data['data'][0]['value'])
        print(f"API로부터 FnG 점수: {score}")
        return score
    except Exception as e:
        print(f"FnG 점수 API 요청 실패: {e}")
        return None

def upload_to_gcs(df, bucket_name, gcs_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    file_exists = blob.exists()

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')

    if file_exists:
        print(f"기존 파일 덮어쓰기 완료: gs://{bucket_name}/{gcs_path}")
    else:
        print(f"새 파일 업로드 완료: gs://{bucket_name}/{gcs_path}")

def find_latest_fng_data(client, bucket_name, prefix):
    """GCS에서 가장 최근의 FNG 데이터 파일 찾기"""
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    date_folders = set()
    
    for blob in blobs:
        parts = blob.name.split('/')
        if len(parts) > 1:
            date_folders.add(parts[1])
    
    sorted_dates = sorted(list(date_folders), reverse=True)
    today = datetime.now().date()
    
    for date_str in sorted_dates:
        try:
            folder_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if folder_date == today:
                continue
            if (today - folder_date).days <= 7:
                path = f"{prefix}/{date_str}/fng_data.csv"
                if bucket.blob(path).exists():
                    return pd.read_csv(io.BytesIO(bucket.blob(path).download_as_bytes()))
        except:
            continue
    return None

if __name__ == '__main__':
    today = datetime.now().strftime('%Y-%m-%d')
    score = fetch_fng_score()

    if score is not None:
        fng_df = pd.DataFrame({
            'date': [today],
            'fear_greed_score': [score]
        })
    else:
        # 실패 시 기본값
        print("FnG 점수를 가져오지 못했습니다. 기본값(50)을 사용합니다.")
        fng_df = pd.DataFrame({
            'date': [today],
            'fear_greed_score': [50]
        })

    gcs_fng_path = f"{GCS_FOLDER_PREFIX}/{today}/fng_data.csv"
    upload_to_gcs(fng_df, BUCKET_NAME, gcs_fng_path)

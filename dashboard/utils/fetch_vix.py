import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import storage
import io

# 설정
BUCKET_NAME = 'emotion-index-data'
GCS_FOLDER_PREFIX = 'VIX_FNG'

def fetch_vix_data(start='2023-01-01'):
    try:
        # VIX 다운로드
        df = yf.download('^VIX', start=start, progress=False)
        
        # 데이터가 비어있는지 확인
        if df.empty:
            print("yfinance에서 VIX 데이터를 가져오지 못했습니다.")
            return None
            
        df = df[['Close']].reset_index()
        df.columns = ['date', 'close']
        df['date'] = df['date'].dt.date
        
        # 최신 데이터만 필요하므로 마지막 행 반환
        latest_data = df.iloc[-1:].copy()
        return latest_data
    except Exception as e:
        print(f"VIX 데이터 다운로드 실패: {e}")
        return None

def find_latest_vix_data(client, bucket_name, prefix):
    """GCS에서 가장 최근의 VIX 데이터 파일 찾기"""
    bucket = client.bucket(bucket_name)
    
    # 날짜별 폴더 리스트 가져오기
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    date_folders = set()
    
    # 날짜 폴더 추출
    for blob in blobs:
        path_parts = blob.name.split('/')
        if len(path_parts) > 1:
            date_folders.add(path_parts[1])
    
    # 날짜 정렬 (최신순)
    sorted_dates = sorted(list(date_folders), reverse=True)
    
    # 최근 7일 내에서 검색
    today = datetime.now().date()
    
    for date_str in sorted_dates:
        try:
            # 날짜 폴더 형식이 맞는지 확인
            folder_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            
            # 오늘 날짜는 제외 (현재 저장하려는 날짜)
            if folder_date == today:
                continue
                
            # 7일 이내의 데이터만 확인
            if (today - folder_date).days <= 7:
                vix_path = f"{prefix}/{date_str}/vix_data.csv"
                blob = bucket.blob(vix_path)
                
                if blob.exists():
                    print(f"최근 VIX 데이터 찾음: {date_str}")
                    content = blob.download_as_bytes()
                    df = pd.read_csv(io.BytesIO(content))
                    return df
        except:
            continue
    
    return None

def upload_to_gcs(df, bucket_name, gcs_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    
    # 파일 존재 여부 확인
    file_exists = blob.exists()
    
    # DataFrame → CSV 메모리 저장
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # GCS 업로드
    blob.upload_from_string(csv_data, content_type='text/csv')
    
    if file_exists:
        print(f"기존 파일 덮어쓰기 완료: gs://{bucket_name}/{gcs_path}")
    else:
        print(f"새 파일 업로드 완료: gs://{bucket_name}/{gcs_path}")

if __name__ == '__main__':
    # 오늘 날짜로 폴더 생성
    today = datetime.now().strftime('%Y-%m-%d')
    vix_df = fetch_vix_data()
    
    client = storage.Client()

    if vix_df is None or vix_df.empty:
        print("오늘의 VIX 데이터를 가져오지 못했습니다. 최근 데이터를 찾습니다.")
        vix_df = find_latest_vix_data(client, BUCKET_NAME, GCS_FOLDER_PREFIX)
        
        if vix_df is not None:
            # 날짜를 오늘로 업데이트
            print(f"최근 VIX 재사용: {vix_df.iloc[0]['close']}")
            vix_df['date'] = today
        else:
            print("최근 VIX 데이터도 찾지 못했습니다. 기본값을 사용합니다.")
            # 기본값 설정 (중립적인 값으로 20으로 설정)
            vix_df = pd.DataFrame({
                'date': [today],
                'close': [20.0]  # 평균적인 VIX 값
            })
            print("기본값(VIX=20.0) 사용")

    # 저장 경로 구성
    gcs_vix_path = f"{GCS_FOLDER_PREFIX}/{today}/vix_data.csv"

    # 업로드
    upload_to_gcs(vix_df, BUCKET_NAME, gcs_vix_path)

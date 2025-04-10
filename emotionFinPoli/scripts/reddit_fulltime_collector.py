import os
import json
import datetime
import logging
import pandas as pd
import praw
from google.cloud import storage
from time import sleep
import sys
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

# ✅ 설정
SUBREDDITS = [
    # 경제 관련
    "economics", "economy", "MacroEconomics", "EconMonitor",
    # 금융 관련
    "finance", "investing", "financialindependence", "personalfinance",
    "wallstreetbets", "stocks", "StockMarket", "dividends",
    # 감정 기반 흐름
    "anxiety", "depression", "offmychest"
]

BUCKET_NAME = "emotion-raw-data"
POST_LIMIT = 1000  # 서브레딧 전체 수집 기준

# ✅ 주의 알림
print("\033[91m⚠️  Reddit API는 최대 1년 이내 게시물만 수집할 수 있습니다.\033[0m")

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reddit_full_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_full_collector")

# ✅ Reddit API 초기화 (.env 파일에서 로드)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# ✅ 저장 함수
def save_to_gcs(df, subreddit, date_str):
    # 모든 서브레딧의 데이터에 서브레딧 정보 추가
    df['subreddit'] = subreddit
    
    # GDELT와 동일한 방식으로 파일명 지정
    file_path = f"sns/reddit/full/{date_str}.json"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    
    # 기존 파일 존재 여부 확인하여 병합
    if blob.exists():
        try:
            existing_data = json.loads(blob.download_as_string())
            combined_data = existing_data + df.to_dict(orient='records')
            blob.upload_from_string(json.dumps(combined_data, ensure_ascii=False), content_type='application/json')
        except Exception as e:
            logger.error(f"기존 파일 병합 중 오류: {e}")
            # 오류 발생 시 새 파일로 저장
            blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    else:
        # 새 파일 생성
        blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
    
    logger.info(f"✅ 저장 완료: gs://{BUCKET_NAME}/{file_path} (서브레딧: {subreddit})")

# ✅ 메인 실행 함수
def run(start_str, end_str):
    # 날짜 범위 파싱
    start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    
    logger.info(f"🚀 Reddit 전체 수집 시작: {start_str} ~ {end_str}")
    logger.info(f"📋 총 {len(SUBREDDITS)}개 서브레딧 수집 예정")
    
    # GDELT와 동일한 방식으로 파일명 생성
    date_range = f"{start_str}_to_{end_str}"
    
    # 전체 진행 상황 추적
    total_subreddits = len(SUBREDDITS)
    total_posts_collected = 0
    start_time = datetime.datetime.now()
    subreddit_results = {}
    
    for idx, sub in enumerate(SUBREDDITS):
        # 서브레딧 진행률 표시
        progress = (idx / total_subreddits) * 100
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        remaining = 0 if idx == 0 else (elapsed / idx) * (total_subreddits - idx)
        
        logger.info(f"🔄 전체 진행률: {progress:.1f}% ({idx+1}/{total_subreddits}) - "
                   f"예상 남은 시간: {remaining/60:.1f}분")
        logger.info(f"📦 서브레딧 수집 시작: r/{sub} ({start_str} ~ {end_str})")
        
        posts = []
        subreddit_start_time = datetime.datetime.now()

        try:
            # 수집 시간 필터 설정
            # Reddit API는 특정 날짜 범위 지정이 제한적이므로 시간 필터로 대략 접근
            time_filter = 'all'  # 기본값
            
            # 현재 시점으로부터 날짜 차이 계산
            days_diff = (datetime.utcnow().date() - start_dt.date()).days
            
            if days_diff <= 1:
                time_filter = 'day'
            elif days_diff <= 7:
                time_filter = 'week'
            elif days_diff <= 30:
                time_filter = 'month'
            elif days_diff <= 90:
                time_filter = 'month'
            elif days_diff <= 365:
                time_filter = 'year'
            else:
                time_filter = 'all'
                logger.warning(f"⚠️ 1년 이상 지난 데이터는 완전히 수집되지 않을 수 있습니다.")
                
            logger.info(f"🕒 시간 필터: {time_filter} (현재 시점부터 {days_diff}일 전)")
            
            # 수집 예상 게시물 수 안내
            logger.info(f"💬 최대 {POST_LIMIT}개 게시물 탐색 예정")
            
            # 데이터 수집
            collected_count = 0
            matched_count = 0
            
            for submission in reddit.subreddit(sub).top(limit=POST_LIMIT, time_filter=time_filter):
                collected_count += 1
                
                # 100개마다 로그 추가 및 예상 시간 계산
                if collected_count % 100 == 0 or collected_count == 1:
                    progress = (collected_count / POST_LIMIT) * 100
                    elapsed_sub = (datetime.datetime.now() - subreddit_start_time).total_seconds()
                    remaining_sub = 0 if collected_count == 0 else (elapsed_sub / collected_count) * (POST_LIMIT - collected_count)
                    
                    logger.info(f"  → 현재 {collected_count}/{POST_LIMIT}개 탐색 중... ({progress:.1f}%)")
                    logger.info(f"  → 해당 서브레딧 예상 남은 시간: {remaining_sub/60:.1f}분")
                    logger.info(f"  → 날짜 범위에 맞는 게시물: {matched_count}개")
                
                # 날짜 범위 필터링
                created_dt = datetime.datetime.utcfromtimestamp(submission.created_utc)
                if start_dt <= created_dt <= end_dt:
                    posts.append({
                        "id": submission.id,
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "created_utc": created_dt.isoformat(),
                        "score": submission.score,
                        "num_comments": submission.num_comments,
                        "url": submission.url
                    })
                    matched_count += 1
                    
                    # 50개마다 날짜 범위 로깅
                    if matched_count % 50 == 0:
                        logger.info(f"  → ✓ {matched_count}개 게시물 수집됨 (날짜 범위 내)")
                        
                sleep(0.5)  # 요청 간 rate limit 방지

            # 서브레딧 수집 완료 상태 저장
            subreddit_elapsed = (datetime.datetime.now() - subreddit_start_time).total_seconds()
            subreddit_results[sub] = {
                "collected": matched_count,
                "searched": collected_count,
                "time_taken": subreddit_elapsed,
                "time_filter": time_filter
            }
            
            logger.info(f"✅ r/{sub} 탐색 완료 ({collected_count}개 중 {matched_count}개 수집, "
                      f"소요 시간: {subreddit_elapsed/60:.1f}분)")

        except Exception as e:
            logger.error(f"🔴 r/{sub} 수집 중 오류 발생: {e}")
            subreddit_results[sub] = {"error": str(e), "collected": 0}
            continue

        df = pd.DataFrame(posts)
        if df.empty:
            logger.warning(f"⚠️ 수집된 글 없음: r/{sub}")
            continue

        total_posts_collected += len(df)
        logger.info(f"📊 r/{sub}에서 {len(df)}개 게시물 수집됨")
        save_to_gcs(df, sub, date_range)

    # 전체 수집 결과 요약
    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*50)
    logger.info(f"📈 Reddit 수집 최종 결과:")
    logger.info(f"  • 수집 기간: {start_str} ~ {end_str}")
    logger.info(f"  • 총 서브레딧 수: {len(SUBREDDITS)}개")
    logger.info(f"  • 총 수집 게시물: {total_posts_collected}개")
    logger.info(f"  • 총 소요 시간: {total_elapsed/60:.1f}분")
    
    logger.info("\n🔍 서브레딧별 수집 상세:")
    for sub, result in subreddit_results.items():
        if "error" in result:
            logger.info(f"  • r/{sub}: 오류 발생 - {result['error']}")
        else:
            success_rate = (result['collected'] / result['searched'] * 100) if result['searched'] > 0 else 0
            logger.info(f"  • r/{sub}: {result['collected']}개 수집 / {result['searched']}개 탐색 "
                      f"({success_rate:.1f}%) - 소요 시간: {result['time_taken']/60:.1f}분")
    
    logger.info("="*50)
    logger.info("🎉 전체 Reddit 수집 완료!")

if __name__ == "__main__":
    # 명령줄 인자 확인
    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
    else:
        # 사용자 입력 받기
        print("Reddit 데이터 수집 도구")
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
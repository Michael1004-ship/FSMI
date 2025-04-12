from datetime import datetime, timedelta
import os
import json
import logging
import pandas as pd
import praw
from google.cloud import storage
from time import sleep
import sys
import argparse
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import random

# ✅ 환경 변수 로드
load_dotenv()

# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 스크립트 이름 기반으로 로그 파일 설정
script_name = os.path.basename(__file__)
log_file = f"{LOG_DATE_DIR}/{script_name.replace('.py', '.log')}"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

# ✅ 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_fulltime_collector")

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

# 병렬 처리 설정
MAX_WORKERS = 5  # 동시 처리할 최대 서브레딧 수
REQUEST_DELAY = 1.0  # 기본 요청 간 지연 시간(초)
MAX_RETRIES = 3  # API 요청 재시도 횟수

# ✅ 주의 알림
logger.warning("\033[91m⚠️  Reddit API는 최대 1년 이내 게시물만 수집할 수 있습니다.\033[0m")

# ✅ Reddit API 초기화 (.env 파일에서 로드)
def init_reddit_api():
    """Reddit API 초기화 (재시도 로직 포함)"""
    for attempt in range(MAX_RETRIES):
        try:
            reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT"),
            )
            # API 접속 테스트
            reddit.user.me()
            return reddit
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep_time = 2 ** attempt  # 지수 백오프
                logger.warning(f"Reddit API 연결 실패 ({attempt + 1}/{MAX_RETRIES}): {e}")
                logger.warning(f"{sleep_time}초 후 재시도...")
                sleep(sleep_time)
            else:
                logger.error(f"Reddit API 연결 실패: {e}")
                raise
    
    return None

# ✅ 저장 함수
def save_to_gcs(df, date_range):
    """데이터 GCS에 저장"""
    # GDELT와 동일한 방식으로 파일명 지정
    file_path = f"sns/reddit/full/{date_range}.json"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_path)
    
    # 기존 파일 존재 여부 확인하여 병합
    if blob.exists():
        try:
            existing_data = json.loads(blob.download_as_string())
            combined_data = existing_data + df.to_dict(orient='records')
            blob.upload_from_string(json.dumps(combined_data, ensure_ascii=False), content_type='application/json')
            logger.info(f"🔄 기존 파일 업데이트 완료: gs://{BUCKET_NAME}/{file_path}")
        except Exception as e:
            logger.error(f"기존 파일 병합 중 오류: {e}")
            # 오류 발생 시 새 파일로 저장
            blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
            logger.info(f"✅ 새 파일 저장 완료: gs://{BUCKET_NAME}/{file_path}")
    else:
        # 새 파일 생성
        blob.upload_from_string(json.dumps(df.to_dict(orient='records'), ensure_ascii=False), content_type='application/json')
        logger.info(f"✅ 새 파일 생성 완료: gs://{BUCKET_NAME}/{file_path}")
    
    return len(df)

# ✅ 서브레딧 데이터 수집 함수 (병렬 처리용)
def collect_subreddit(reddit, subreddit_name, start_dt, end_dt, time_filter):
    """단일 서브레딧의 데이터 수집 (병렬 처리용)"""
    posts = []
    collected_count = 0
    matched_count = 0
    error = None
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 r/{subreddit_name} 수집 시작 (시간 필터: {time_filter})")
        
        for submission in reddit.subreddit(subreddit_name).top(limit=POST_LIMIT, time_filter=time_filter):
            collected_count += 1
            
            # 요청 간 지연
            delay = REQUEST_DELAY + random.uniform(0, 0.5)  # 0.5초 랜덤 추가
            sleep(delay)
            
            # 날짜 범위 필터링
            created_dt = datetime.utcfromtimestamp(submission.created_utc)
            if start_dt <= created_dt <= end_dt:
                posts.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "url": submission.url,
                    "subreddit": subreddit_name
                })
                matched_count += 1
                
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "subreddit": subreddit_name,
            "collected": collected_count,
            "matched": matched_count,
            "posts": posts,
            "time_taken": elapsed,
            "success": True,
            "time_filter": time_filter
        }
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"r/{subreddit_name} 수집 중 오류: {e}")
        return {
            "subreddit": subreddit_name,
            "collected": collected_count,
            "matched": matched_count,
            "posts": posts,
            "time_taken": elapsed,
            "success": False,
            "error": str(e),
            "time_filter": time_filter
        }

def run(start_str, end_str, max_workers=MAX_WORKERS, debug=False):
    """메인 실행 함수"""
    # 날짜 범위 파싱
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    
    logger.info(f"🚀 Reddit 전체 수집 시작: {start_str} ~ {end_str}")
    logger.info(f"📋 총 {len(SUBREDDITS)}개 서브레딧 수집 예정")
    logger.info(f"🧵 병렬 처리 스레드 수: {max_workers}")
    
    # GDELT와 동일한 방식으로 파일명 생성
    date_range = f"{start_str}_to_{end_str}"
    
    # 전체 진행 상황 추적
    total_subreddits = len(SUBREDDITS)
    total_posts_collected = 0
    start_time = datetime.now()
    subreddit_results = {}
    
    # Reddit API 초기화
    try:
        reddit = init_reddit_api()
        if not reddit:
            logger.error("❌ Reddit API 초기화 실패")
            return
    except Exception as e:
        logger.error(f"❌ Reddit API 초기화 실패: {e}")
        return
    
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
    
    logger.info(f"🕒 선택된 시간 필터: {time_filter} (현재 시점부터 {days_diff}일 전)")
    
    # 병렬 처리를 통한 서브레딧 데이터 수집
    all_posts = []
    failed_subreddits = []
    
    with tqdm(total=len(SUBREDDITS), desc="서브레딧 수집 중", unit="개") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 서브레딧별 작업 제출
            future_to_subreddit = {
                executor.submit(collect_subreddit, reddit, sub, start_dt, end_dt, time_filter): sub 
                for sub in SUBREDDITS
            }
            
            # 결과 처리
            for future in concurrent.futures.as_completed(future_to_subreddit):
                subreddit = future_to_subreddit[future]
                try:
                    result = future.result()
                    subreddit_results[subreddit] = result
                    
                    if result["success"]:
                        all_posts.extend(result["posts"])
                        logger.info(f"✅ r/{subreddit} 수집 완료: 검색 {result['collected']}개 중 {result['matched']}개 수집")
                    else:
                        failed_subreddits.append(subreddit)
                        logger.error(f"❌ r/{subreddit} 수집 실패: {result.get('error', '알 수 없는 오류')}")
                except Exception as e:
                    subreddit_results[subreddit] = {
                        "success": False,
                        "error": str(e)
                    }
                    failed_subreddits.append(subreddit)
                    logger.error(f"❌ r/{subreddit} 처리 중 예외 발생: {str(e)}")
                
                # 진행 상황 업데이트
                pbar.update(1)
                
                # 주기적 상태 보고
                if pbar.n % max(1, len(SUBREDDITS) // 5) == 0 or pbar.n == len(SUBREDDITS):
                    elapsed = (datetime.now() - start_time).total_seconds()
                    remaining = (elapsed / pbar.n) * (len(SUBREDDITS) - pbar.n) if pbar.n > 0 else 0
                    success_count = len([r for r in subreddit_results.values() if r.get("success", False)])
                    total_collected = sum([r.get("matched", 0) for r in subreddit_results.values()])
                    
                    logger.info(f"🔄 진행률: {pbar.n/len(SUBREDDITS)*100:.1f}% ({pbar.n}/{len(SUBREDDITS)}) - "
                               f"성공: {success_count}, 실패: {len(failed_subreddits)} - "
                               f"수집 게시물: {total_collected}개 - "
                               f"예상 남은 시간: {remaining/60:.1f}분")
    
    # 결과 처리
    if all_posts:
        # DataFrame 생성
        df = pd.DataFrame(all_posts)
        
        # 중복 제거
        df = df.drop_duplicates(subset=["id"])
        
        # GCS에 저장
        total_saved = save_to_gcs(df, date_range)
        total_posts_collected = len(df)
        
        logger.info(f"📊 총 {total_posts_collected}개 게시물 수집됨 (중복 제거 후)")
    else:
        logger.warning("⚠️ 수집된 게시물이 없습니다.")
    
    # 전체 수집 결과 요약
    total_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*50)
    logger.info(f"📈 Reddit 수집 최종 결과:")
    logger.info(f"  • 수집 기간: {start_str} ~ {end_str}")
    logger.info(f"  • 총 서브레딧 수: {len(SUBREDDITS)}개")
    logger.info(f"  • 성공한 서브레딧: {len(SUBREDDITS) - len(failed_subreddits)}개")
    logger.info(f"  • 실패한 서브레딧: {len(failed_subreddits)}개")
    logger.info(f"  • 총 수집 게시물: {total_posts_collected}개")
    logger.info(f"  • 총 소요 시간: {total_elapsed/60:.1f}분")
    
    # 서브레딧별 수집 상세
    logger.info("\n🔍 서브레딧별 수집 상세:")
    for sub, result in subreddit_results.items():
        if not result.get("success", False):
            logger.info(f"  • r/{sub}: 오류 발생 - {result.get('error', '알 수 없는 오류')}")
        else:
            success_rate = (result['matched'] / result['collected'] * 100) if result['collected'] > 0 else 0
            logger.info(f"  • r/{sub}: {result['matched']}개 수집 / {result['collected']}개 탐색 "
                      f"({success_rate:.1f}%) - 소요 시간: {result['time_taken']/60:.1f}분")
    
    # 실패한 서브레딧 로그
    if failed_subreddits:
        fail_log_path = f"{LOG_DATE_DIR}/reddit_failed_subreddits_{date_range}.json"
        with open(fail_log_path, 'w') as f:
            failures = {sub: subreddit_results.get(sub, {"error": "No data"}) for sub in failed_subreddits}
            json.dump(failures, f, indent=2)
        logger.info(f"📝 실패 서브레딧 로그 저장됨: {fail_log_path}")
    
    logger.info("="*50)
    logger.info("🎉 전체 Reddit 수집 완료!")
    
    return total_posts_collected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit 데이터 수집기 (병렬 처리)")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"병렬 처리 스레드 수 (기본값: {MAX_WORKERS})")
    args = parser.parse_args()
    
    try:
        # 날짜 형식 검증
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        
        if end_date < start_date:
            logger.error("❌ 종료일이 시작일보다 앞설 수 없습니다.")
            sys.exit(1)
            
        logger.info(f"📅 수집 기간: {args.start} ~ {args.end}")
        
        # Reddit API 연결 확인
        if not all([os.getenv("REDDIT_CLIENT_ID"), 
                   os.getenv("REDDIT_CLIENT_SECRET"), 
                   os.getenv("REDDIT_USER_AGENT")]):
            logger.error("❌ Reddit API 인증 정보가 설정되지 않았습니다.")
            sys.exit(1)
        
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.info("🔧 디버그 모드 활성화됨")
        
        # 실행
        run(args.start, args.end, args.workers, args.debug)
        
    except ValueError as e:
        logger.error(f"❌ 날짜 형식이 올바르지 않습니다: {str(e)}")
        sys.exit(1)
    except Exception as e:
        if args.debug:
            logger.exception("오류 발생:")
        else:
            logger.error(f"오류 발생: {str(e)}")
        sys.exit(1)
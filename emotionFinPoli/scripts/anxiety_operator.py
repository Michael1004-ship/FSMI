# emotion_data_processor.py
import subprocess
import logging
import datetime
import time
import os
import sys
import traceback
import platform
import psutil
from google.cloud import storage
import json

# 상세 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_data_processor_detailed.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotion_data_processor")

# 스크립트 경로 설정
SCRIPTS_DIR = "scripts"
BUILD_INDEX_DIR = "building_index"
BUCKET_RAW = "emotion-raw-data"
BUCKET_INDEX = "emotion-index-data"

# 파일 경로를 절대 경로로 직접 설정
GDELT_COLLECTOR = "/home/hwangjeongmun691/projects/emotionFinPoli/scripts/gdelt_realtime_collector.py"
GDELT_ANALYSIS = "/home/hwangjeongmun691/projects/emotionFinPoli/scripts/gdelt_realtime_crawling&FinBERT.py"
REDDIT_COLLECTOR = "/home/hwangjeongmun691/projects/emotionFinPoli/scripts/reddit_realtime_collector.py"
REDDIT_FINBERT = "/home/hwangjeongmun691/projects/emotionFinPoli/scripts/reddit_FinBERT.py"
REDDIT_ROBERTA = "/home/hwangjeongmun691/projects/emotionFinPoli/scripts/reddit_RoBERTa.py"
BUILD_INDEX = "/home/hwangjeongmun691/projects/emotionFinPoli/building_index/build_anxiety_index.py"

def check_environment():
    """시스템 환경 정보를 로깅합니다."""
    logger.info("===== 시스템 환경 정보 =====")
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"운영체제: {platform.platform()}")
    logger.info(f"현재 작업 디렉토리: {os.getcwd()}")
    logger.info(f"CPU 코어: {psutil.cpu_count()} 개")
    
    memory = psutil.virtual_memory()
    logger.info(f"메모리: 전체 {memory.total/1024/1024/1024:.1f}GB, 사용 가능 {memory.available/1024/1024/1024:.1f}GB")
    
    # GCS 연결 확인
    try:
        client = storage.Client()
        logger.info(f"GCS 연결 확인: 성공")
        # 버킷 존재 확인
        for bucket_name in [BUCKET_RAW, BUCKET_INDEX]:
            if client.bucket(bucket_name).exists():
                logger.info(f"버킷 확인: {bucket_name} ✓")
            else:
                logger.error(f"버킷 확인: {bucket_name} ✗")
    except Exception as e:
        logger.error(f"GCS 연결 실패: {e}")

def check_data_files(date_str):
    """특정 날짜의 데이터 파일 존재 여부 확인"""
    client = storage.Client()
    
    # GDELT 데이터 확인
    gdelt_path = f"news/gdelt/{date_str}/accumulated.json"
    if client.bucket(BUCKET_RAW).blob(gdelt_path).exists():
        logger.info(f"GDELT 데이터 확인: {gdelt_path} ✓")
    else:
        logger.warning(f"GDELT 데이터 확인: {gdelt_path} ✗")
    
    # Reddit 데이터 확인 (첫 번째 서브레딧만)
    subreddits = ["anxiety", "stocks", "economics"]
    for sub in subreddits:
        reddit_path = f"sns/reddit/{sub}/{date_str}/accumulated.json"
        if client.bucket(BUCKET_RAW).blob(reddit_path).exists():
            logger.info(f"Reddit 데이터 확인 ({sub}): ✓")
            break
    else:
        logger.warning(f"Reddit 데이터 확인: 모든 서브레딧 ✗")
    
    # 분석 결과 파일 확인
    files_to_check = [
        (BUCKET_INDEX, f"news/{date_str}/news_anxiety_index.csv"),
        (BUCKET_INDEX, f"reddit/{date_str}/reddit_anxiety_index.csv"),
        (BUCKET_INDEX, f"reddit/{date_str}/reddit_anxiety_roberta.csv")
    ]
    
    for bucket_name, path in files_to_check:
        if client.bucket(bucket_name).blob(path).exists():
            logger.info(f"분석 결과 확인: {path} ✓")
        else:
            logger.warning(f"분석 결과 확인: {path} ✗")

def run_script(script_path, description, args=None, retry=1):
    """스크립트를 실행하고 상세 디버깅 정보를 로깅합니다."""
    for attempt in range(1, retry+1):
        try:
            start_time = time.time()
            memory_before = psutil.virtual_memory().percent
            
            logger.info(f"✨ {description} 시작 (시도 {attempt}/{retry})...")
            logger.debug(f"실행 명령: {sys.executable} {script_path} {args if args else ''}")
            
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
                
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            elapsed_time = time.time() - start_time
            memory_after = psutil.virtual_memory().percent
            memory_diff = memory_after - memory_before
            
            logger.info(f"✅ {description} 완료 (소요시간: {elapsed_time:.2f}초)")
            logger.debug(f"메모리 사용 변화: {memory_before}% → {memory_after}% (차이: {memory_diff:.1f}%)")
            
            if result.stdout:
                logger.debug("===== 스크립트 표준 출력 =====")
                for line in result.stdout.splitlines():
                    if "error" in line.lower() or "exception" in line.lower():
                        logger.warning(line)
                    elif "warn" in line.lower():
                        logger.warning(line)
                    else:
                        logger.debug(line)
            
            if result.stderr:
                logger.warning("===== 스크립트 오류 출력 =====")
                for line in result.stderr.splitlines():
                    logger.warning(line)
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} 실패 (시도 {attempt}/{retry}): 종료 코드 {e.returncode}")
            
            if e.stdout:
                logger.error("===== 스크립트 표준 출력 =====")
                for line in e.stdout.splitlines()[-20:]:  # 마지막 20줄만 출력
                    logger.error(line)
            
            if e.stderr:
                logger.error("===== 스크립트 오류 출력 =====")
                for line in e.stderr.splitlines():
                    logger.error(line)
            
            if attempt < retry:
                logger.info(f"⏱️ 5초 후 재시도...")
                time.sleep(5)
            else:
                return False
        
        except Exception as e:
            logger.error(f"❌ {description} 예외 발생 (시도 {attempt}/{retry}): {e}")
            logger.error(traceback.format_exc())
            
            if attempt < retry:
                logger.info(f"⏱️ 5초 후 재시도...")
                time.sleep(5)
            else:
                return False

def main():
    start_time = time.time()
    logger.info("🚀 감정 데이터 처리 파이프라인 시작")
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    
    # 시스템 환경 확인
    check_environment()
    
    # 스크립트 파일 존재 확인
    for path, name in [
        (GDELT_COLLECTOR, "GDELT 수집기"),
        (GDELT_ANALYSIS, "GDELT 분석기"),
        (REDDIT_COLLECTOR, "Reddit 수집기"),
        (REDDIT_FINBERT, "Reddit FinBERT"),
        (REDDIT_ROBERTA, "Reddit RoBERTa"),
        (BUILD_INDEX, "불안 지수 빌더")
    ]:
        if os.path.exists(path):
            logger.info(f"✓ {name} 스크립트 확인됨")
        else:
            logger.error(f"✗ {name} 스크립트 없음! 경로: {path}")
    
    # 1. 데이터 수집 단계
    gdelt_success = run_script(GDELT_COLLECTOR, "GDELT 데이터 수집", retry=2)
    if not gdelt_success:
        logger.error("❗ GDELT 데이터 수집 실패, 계속 진행합니다.")
    
    reddit_success = run_script(REDDIT_COLLECTOR, "Reddit 데이터 수집", retry=2)
    if not reddit_success:
        logger.error("❗ Reddit 데이터 수집 실패, 계속 진행합니다.")
    
    # 데이터 파일 존재 확인
    logger.info("===== 데이터 수집 결과 확인 =====")
    check_data_files(today)
    
    # 2. 감정 분석 단계
    gdelt_analysis_success = run_script(GDELT_ANALYSIS, "GDELT 기사 크롤링 및 FinBERT 분석", retry=2)
    if not gdelt_analysis_success:
        logger.error("❗ GDELT 분석 실패, 계속 진행합니다.")
    
    reddit_finbert_success = run_script(REDDIT_FINBERT, "Reddit FinBERT 분석", retry=2)
    if not reddit_finbert_success:
        logger.error("❗ Reddit FinBERT 분석 실패, 계속 진행합니다.")
    
    reddit_roberta_success = run_script(REDDIT_ROBERTA, "Reddit RoBERTa 분석", retry=2)
    if not reddit_roberta_success:
        logger.error("❗ Reddit RoBERTa 분석 실패, 계속 진행합니다.")
    
    # 3. 지수 통합 단계
    index_success = run_script(BUILD_INDEX, "불안 지수 통합", ["--date", today], retry=2)
    if not index_success:
        logger.error("❗ 불안 지수 통합 실패")
    
    elapsed_time = time.time() - start_time
    logger.info(f"🏁 감정 데이터 처리 파이프라인 완료 (전체 소요시간: {elapsed_time/60:.2f}분)")
    
    # 실행 결과 요약
    logger.info("===== 실행 결과 요약 =====")
    for step, success in [
        ("GDELT 데이터 수집", gdelt_success),
        ("Reddit 데이터 수집", reddit_success),
        ("GDELT 분석", gdelt_analysis_success),
        ("Reddit FinBERT 분석", reddit_finbert_success),
        ("Reddit RoBERTa 분석", reddit_roberta_success),
        ("불안 지수 통합", index_success)
    ]:
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"{step}: {status}")

if __name__ == "__main__":
    main()
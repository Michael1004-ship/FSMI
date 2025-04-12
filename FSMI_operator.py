# main_operator.py
import subprocess
import argparse
import logging
import time
import sys
import os
from datetime import datetime

# 디버깅용 정보 출력
print("✅ 현재 파이썬:", sys.executable)
print("✅ 현재 작업 디렉토리:", os.getcwd())

# 로깅 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"
current_time = datetime.utcnow().strftime("%H%M%S")
log_filename = f"fsmi_operator_{today.replace('-', '')}_{current_time}.log"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/{log_filename}"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FSMI_operator")
logger.info(f"📋 로그 저장 위치: {LOG_DATE_DIR}/{log_filename}")

# 가상환경 Python 경로 설정
VENV_PYTHON = "/home/hwangjeongmun691/projects/emotionFinPoli/env/bin/python"

def run_script(path, debug=False, retry=3):
    """스크립트를 실행하고 디버깅 정보를 제공합니다."""
    script_name = os.path.basename(path)
    header = f"\n{'='*30} 실행: {script_name} {'='*30}"
    logger.info(header)
    
    start_time = time.time()
    success = False
    attempt = 0
    
    while not success and attempt < retry:
        attempt += 1
        if attempt > 1:
            logger.warning(f"재시도 {attempt}/{retry}: {script_name}")
        
        try:
            # 명령 구성 - 가상환경 Python 사용
            cmd = [VENV_PYTHON, path]
            if debug:
                cmd.append("--debug")
            
            # 실행 전 메모리 사용량
            import psutil
            process = psutil.Process(os.getpid())
            before_mem = process.memory_info().rss / 1024 / 1024  # MB
            
            # 스크립트 실행
            logger.info(f"🚀 실행 중: {path} (시도 {attempt}/{retry}) - 가상환경: {VENV_PYTHON}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # 출력 로깅
            if result.stdout:
                logger.info(f"스크립트 출력:\n{result.stdout}")
            
            # 성공 시 루프 종료
            success = True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 오류 발생: {path} (종료 코드: {e.returncode})")
            if e.stdout:
                logger.info(f"표준 출력:\n{e.stdout}")
            if e.stderr:
                logger.error(f"오류 출력:\n{e.stderr}")
            
            if attempt == retry:
                logger.critical(f"최대 재시도 횟수 도달: {path}")
                if not debug:
                    logger.info("디버그 모드로 재실행하려면: python FSMI_operator.py --debug --step={스크립트 번호}")
                    raise  # 디버그 모드가 아니면 예외 전달
            
            # 재시도 전 대기
            wait_time = 5 * attempt
            logger.info(f"{wait_time}초 후 재시도합니다...")
            time.sleep(wait_time)
    
    # 실행 후 메모리 사용량
    after_mem = process.memory_info().rss / 1024 / 1024  # MB
    mem_diff = after_mem - before_mem
    
    # 실행 시간 계산
    elapsed_time = time.time() - start_time
    
    # 요약 정보
    footer = f"\n{'='*30} 완료: {script_name} ({'성공' if success else '실패'}) {'='*30}"
    logger.info(footer)
    logger.info(f"⏱️ 실행 시간: {elapsed_time:.2f}초")
    logger.info(f"📊 메모리 변화: {mem_diff:.2f}MB (이전: {before_mem:.2f}MB, 이후: {after_mem:.2f}MB)")
    logger.info(f"📋 로그 저장 위치: {LOG_DATE_DIR}/{log_filename}")
    
    return success

def main():
    parser = argparse.ArgumentParser(description="FSMI 통합 실행기")
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    parser.add_argument('--step', type=int, help='특정 단계만 실행 (1-6)')
    parser.add_argument('--from-step', type=int, help='특정 단계부터 실행')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='로깅 레벨')
    args = parser.parse_args()
    
    # 로그 레벨 설정
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level))
    
    # 디버그 모드 알림
    if args.debug:
        logger.info("🔍 디버그 모드가 활성화되었습니다")
    
    base = "/home/hwangjeongmun691/projects/emotionFinPoli"
    
    # 스크립트 정의 - [단계, 설명, 경로]
    scripts = [
        # 1단계: 데이터 수집
        [1, "GDELT 데이터 수집", f"{base}/scripts/gdelt_realtime_collector.py"],
        [1, "Reddit 데이터 수집", f"{base}/scripts/reddit_realtime_collector.py"],
        
        # 2단계: 점수화
        [2, "GDELT FinBERT 처리", f"{base}/scripts/gdelt_realtime_crawling&FinBERT.py"],
        [2, "Reddit FinBERT 처리", f"{base}/scripts/reddit_FinBERT.py"],
        [2, "Reddit RoBERTa 처리", f"{base}/scripts/reddit_RoBERTa.py"],
        
        # 3단계: 통합 지수화
        [3, "불안 지수 생성", f"{base}/building_index/build_anxiety_index.py"],
        
        # 4단계: 클러스터링 전처리
        [4, "GDELT 전처리", f"{base}/clustering/gdelt_preprocessor.py"],
        [4, "Reddit 전처리", f"{base}/clustering/reddit_preprocessor.py"],
        
        # 5단계: 클러스터링
        [5, "GDELT 클러스터링", f"{base}/clustering/gdelt_clustering.py"],
        [5, "Reddit 클러스터링", f"{base}/clustering/reddit_clustering.py"],
        
        # 6단계: GPT 보고서 작성
        [6, "GPT 라벨링", f"{base}/GPT/gpt_labeling.py"],
        [6, "클러스터 시각화", f"{base}/GPT/visualize_clusters.py"],
        [6, "GPT 보고서 생성", f"{base}/GPT/gpt_report.py"]
    ]
    
    total_start_time = time.time()
    
    # 실행할 스크립트 결정
    if args.step:
        filtered_scripts = [s for s in scripts if s[0] == args.step]
        logger.info(f"🔧 {args.step}단계 스크립트만 실행합니다 ({len(filtered_scripts)}개)")
    elif args.from_step:
        filtered_scripts = [s for s in scripts if s[0] >= args.from_step]
        logger.info(f"🔧 {args.from_step}단계부터 실행합니다 ({len(filtered_scripts)}개)")
    else:
        filtered_scripts = scripts
        logger.info(f"🔧 전체 파이프라인을 실행합니다 ({len(filtered_scripts)}개)")
    
    # 스크립트 실행
    success_count = 0
    total_scripts = len(filtered_scripts)
    
    for i, (step, desc, path) in enumerate(filtered_scripts, 1):
        logger.info(f"진행률: {i}/{total_scripts} ({i/total_scripts*100:.1f}%) - 단계 {step}: {desc}")
        
        if run_script(path, debug=args.debug):
            success_count += 1
    
    # 실행 요약
    total_elapsed = time.time() - total_start_time
    logger.info("\n" + "="*60)
    logger.info(f"📋 실행 요약: {success_count}/{total_scripts} 성공 ({success_count/total_scripts*100:.1f}%)")
    logger.info(f"⏱️ 총 실행 시간: {total_elapsed:.2f}초 ({total_elapsed/60:.2f}분)")
    logger.info(f"📊 평균 스크립트 실행 시간: {total_elapsed/total_scripts:.2f}초")
    logger.info(f"📜 자세한 로그: {LOG_DATE_DIR}/{log_filename}")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"⛔ 치명적 오류 발생: {str(e)}", exc_info=True)
        sys.exit(2)

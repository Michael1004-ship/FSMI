#!/usr/bin/env python3
# 파일명: reddit_parallel_executor.py
from datetime import datetime
import subprocess
import sys
import argparse
import threading
import time
import os
import logging

# 로그 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"
os.makedirs(LOG_DATE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/reddit_parallel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_parallel")

def run_script(script_path, start_date, end_date, workers, debug, prefix):
    """스크립트를 실행하고 출력을 실시간으로 로깅"""
    # 실행 명령 구성
    cmd = [
        "python",
        script_path,
        "--start", start_date,
        "--end", end_date,
        "--workers", str(workers)
    ]
    
    if debug:
        cmd.append("--debug")
    
    logger.info(f"{prefix} 실행 시작: {' '.join(cmd)}")
    start_time = time.time()
    
    # 프로세스 실행
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    # 출력 및 오류 스트림 처리
    def handle_output(stream, log_func, prefix):
        for line in iter(stream.readline, ''):
            if line.strip():
                log_func(f"{prefix} {line.strip()}")
    
    # 출력 스트림 스레드
    stdout_thread = threading.Thread(
        target=handle_output, 
        args=(process.stdout, logger.info, prefix)
    )
    stderr_thread = threading.Thread(
        target=handle_output, 
        args=(process.stderr, logger.error, prefix)
    )
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    # 프로세스 완료 대기
    return_code = process.wait()
    
    # 스레드 종료 대기
    stdout_thread.join()
    stderr_thread.join()
    
    elapsed = time.time() - start_time
    
    if return_code == 0:
        logger.info(f"{prefix} 실행 완료 (코드: {return_code}, 소요시간: {elapsed:.1f}초)")
        return True
    else:
        logger.error(f"{prefix} 실행 실패 (코드: {return_code}, 소요시간: {elapsed:.1f}초)")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reddit FinBERT/RoBERTa 병렬 실행기")
    parser.add_argument("--start", required=True, help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--workers", type=int, default=4, help="스크립트별 병렬 처리 스레드 수 (기본값: 4)")
    args = parser.parse_args()
    
    # 스크립트 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    finbert_script = os.path.join(script_dir, "reddit_fulltime_FinBERT.py")
    roberta_script = os.path.join(script_dir, "reddit_fulltime_RoBERTa.py")
    
    # 스크립트 존재 확인
    if not os.path.exists(finbert_script):
        logger.error(f"❌ FinBERT 스크립트를 찾을 수 없습니다: {finbert_script}")
        return 1
    if not os.path.exists(roberta_script):
        logger.error(f"❌ RoBERTa 스크립트를 찾을 수 없습니다: {roberta_script}")
        return 1
    
    logger.info(f"🚀 Reddit 감정 분석 병렬 실행 시작")
    logger.info(f"📅 처리 기간: {args.start} ~ {args.end}")
    logger.info(f"🧵 스크립트별 병렬 처리 스레드 수: {args.workers}")
    
    start_time = time.time()
    
    # 스레드 생성 및 시작
    finbert_thread = threading.Thread(
        target=run_script,
        args=(finbert_script, args.start, args.end, args.workers, args.debug, "[FinBERT]")
    )
    
    roberta_thread = threading.Thread(
        target=run_script,
        args=(roberta_script, args.start, args.end, args.workers, args.debug, "[RoBERTa]")
    )
    
    finbert_thread.start()
    roberta_thread.start()
    
    # 스레드 완료 대기
    finbert_thread.join()
    roberta_thread.join()
    
    # 실행 시간 계산
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ 병렬 처리 완료! 총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    logger.info(f"{'='*50}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 예기치 않은 오류 발생: {str(e)}")
        logger.exception("상세 오류:")
        sys.exit(1)
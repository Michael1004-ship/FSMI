from datetime import datetime, timedelta
import subprocess
import logging
import os
import sys
import time
import argparse
import json
import threading
import psutil  # 시스템 모니터링용 (pip install psutil 필요)
from tqdm import tqdm  # 진행 표시줄 (pip install tqdm 필요)

# 로그 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"
os.makedirs(LOG_DATE_DIR, exist_ok=True)

# 상태 파일 경로
STATUS_FILE = f"{LOG_DATE_DIR}/pipeline_status.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/emotion_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("emotion_pipeline")

# 전역 상태 변수
pipeline_status = {
    "start_time": None,
    "current_script": None,
    "completed_scripts": [],
    "failed_scripts": [],
    "progress": 0,
    "status": "idle",
    "estimated_completion": None,
    "system_info": {},
    "last_updated": None
}

def update_status():
    """상태 파일을 업데이트합니다."""
    pipeline_status["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_status["system_info"] = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2)
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(pipeline_status, f, indent=2)

def status_monitor(interval=5):
    """주기적으로 상태를 업데이트하는 모니터링 스레드"""
    while pipeline_status["status"] != "completed" and pipeline_status["status"] != "failed":
        update_status()
        time.sleep(interval)

def run_script(script_name, start_date, end_date, debug=False):
    """스크립트 실행 함수"""
    try:
        # 상태 업데이트
        pipeline_status["current_script"] = script_name
        pipeline_status["status"] = "running"
        update_status()
        
        # 스크립트 파일명만 추출 (경로 제외)
        script_basename = os.path.basename(script_name)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 {script_basename} 실행 시작")
        logger.info(f"📅 처리 기간: {start_date} ~ {end_date}")
        
        start_time = time.time()
        
        # 스크립트 전체 경로 설정
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        
        # 로그 디렉토리 생성
        script_log_dir = f"{LOG_ROOT}/{today}"
        os.makedirs(script_log_dir, exist_ok=True)
        
        # 스크립트 로그 경로 설정
        script_log_path = f"{script_log_dir}/{script_basename.replace('.py', '.log')}"
        
        # 환경변수로 로그 파일 전달
        env = os.environ.copy()
        env["EMOTION_LOG_FILE"] = script_log_path
        
        # 파이썬 경로 지정
        python_path = "/home/hwangjeongmun691/projects/emotionFinPoli/env/bin/python3"
        
        # 명령어 구성
        cmd = [
            python_path,
            script_path,
            "--start", start_date,
            "--end", end_date
        ]
        
        if debug:
            cmd.append("--debug")
        
        # 애니메이션 진행 표시줄 (tqdm)
        pbar = None
        if not debug:
            # total 값을 명시적으로 설정
            pbar = tqdm(total=100, desc=f"실행 중: {script_basename}", unit="s")
            
        # 스크립트 실행
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(script_path),
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # 실시간 출력 처리
        stdout_lines = []
        stderr_lines = []
        
        # 실시간 출력 읽기 함수
        def read_output(pipe, storage):
            for line in iter(pipe.readline, ''):
                storage.append(line)
                if debug or "ERROR" in line or "WARNING" in line or "✅" in line:
                    logger.info(f"  {line.strip()}")
                if pbar:
                    # 진행 표시만 업데이트
                    pbar.set_description(f"처리 중: {len(storage)}줄")
                    pbar.refresh()
                
                # 진행률 표시가 있는 경우 상태 업데이트
                if "진행률" in line or "progress" in line.lower():
                    try:
                        progress_part = line.split("%")[0].split(":")[-1].strip()
                        progress = float(progress_part)
                        pipeline_status["progress"] = progress
                        update_status()
                    except:
                        pass
                        
        # 출력 읽기 스레드 시작
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_lines))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # 프로세스 완료 대기
        return_code = process.wait()
        
        # 스레드 종료 대기
        stdout_thread.join()
        stderr_thread.join()
        
        # 진행 표시줄 닫기
        if pbar:
            pbar.close()
        
        # 실행 시간 계산
        elapsed_time = time.time() - start_time
        
        # 성공 여부 확인
        if return_code == 0:
            logger.info(f"✅ {script_basename} 실행 완료")
            logger.info(f"⏱️ 소요 시간: {elapsed_time/60:.1f}분")
            pipeline_status["completed_scripts"].append(script_name)
            pipeline_status["status"] = "success"
            update_status()
            return True
        else:
            logger.error(f"❌ {script_basename} 실행 실패 (코드: {return_code})")
            logger.error(f"⏱️ 소요 시간: {elapsed_time/60:.1f}분")
            if stderr_lines:
                logger.error("오류 출력:")
                for line in stderr_lines[-10:]:  # 마지막 10줄만 출력
                    logger.error(f"  {line.strip()}")
            pipeline_status["failed_scripts"].append(script_name)
            pipeline_status["status"] = "error"
            update_status()
            return False
        
    except Exception as e:
        logger.error(f"❌ {script_basename} 실행 중 예외 발생: {str(e)}")
        if debug:
            logger.exception("상세 오류:")
        pipeline_status["failed_scripts"].append(script_name)
        pipeline_status["status"] = "error"
        update_status()
        return False

def validate_date(date_str):
    """날짜 유효성 검증"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def main():
    parser = argparse.ArgumentParser(description="감정 분석 파이프라인 실행기")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    parser.add_argument("--skip", nargs='+', help="건너뛸 스크립트 이름들")
    parser.add_argument("--monitor", action="store_true", help="시스템 모니터링 활성화")
    args = parser.parse_args()

    # 초기 상태 설정
    pipeline_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_status["status"] = "starting"
    update_status()
    
    # 모니터링 스레드 시작
    if args.monitor:
        monitor_thread = threading.Thread(target=status_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info("🔍 시스템 모니터링이 활성화되었습니다.")

    # 디버그 모드 안내
    if args.debug:
        logger.info("🔧 디버그 모드가 활성화되었습니다.")
    
    # 날짜 입력 처리
    start = args.start
    end = args.end
    
    if not start or not end:
        print("\n감정 분석 파이프라인 실행기")
        print("=" * 30)
        
        while True:
            start = input("시작일 (YYYY-MM-DD 형식): ")
            if validate_date(start):
                break
            print("❌ 올바른 날짜 형식이 아닙니다. YYYY-MM-DD 형식으로 입력해주세요.")
        
        while True:
            end = input("종료일 (YYYY-MM-DD 형식): ")
            end_date = validate_date(end)
            start_date = validate_date(start)
            if not end_date:
                print("❌ 올바른 날짜 형식이 아닙니다. YYYY-MM-DD 형식으로 입력해주세요.")
                continue
            if end_date < start_date:
                print("❌ 종료일이 시작일보다 앞설 수 없습니다.")
                continue
            break

    logger.info(f"\n📋 감정 분석 파이프라인 시작")
    logger.info(f"📅 처리 기간: {start} ~ {end}")
    
    # 병렬 실행을 구현하는 코드 추가
    import threading

    # 병렬 실행할 스크립트들 그룹화
    parallel_scripts = {
        "group1": ["reddit_fulltime_FinBERT.py", "reddit_fulltime_RoBERTa.py"]
    }

    # 스크립트 목록 수정
    scripts = [
        "gdelt_fulltime_collector.py",
        "reddit_fulltime_collector.py",
        "gdelt_fulltime_crawling&FinBERT.py",
        "parallel_group1",  # FinBERT와 RoBERTa를 병렬로 실행
        "build_anxiety_index.py"
    ]
    
    # 건너뛸 스크립트 처리
    if args.skip:
        skipped = [s for s in args.skip if s in scripts]
        if skipped:
            logger.info(f"⏭️ 다음 스크립트들을 건너뜁니다: {', '.join(skipped)}")
            scripts = [s for s in scripts if s not in skipped]
    
    total_start_time = time.time()
    success_count = 0
    failed_scripts = []
    
    # 상태 파일 초기화
    pipeline_status["total_scripts"] = len(scripts)
    pipeline_status["status"] = "running"
    update_status()
    
    # 각 스크립트 순차 실행
    for i, script in enumerate(scripts, 1):
        # 진행률 계산 및 표시
        progress_percent = (i - 1) / len(scripts) * 100
        remaining_scripts = len(scripts) - (i - 1)
        
        logger.info(f"\n[{i}/{len(scripts)}] {script} 실행 중... (전체 진행률: {progress_percent:.1f}%)")
        
        # 예상 완료 시간 계산 및 업데이트
        if i > 1 and success_count > 0:
            elapsed_so_far = time.time() - total_start_time
            avg_time_per_script = elapsed_so_far / (i - 1)
            estimated_remaining = avg_time_per_script * remaining_scripts
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining)
            
            pipeline_status["estimated_completion"] = estimated_completion.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"⏱️ 예상 남은 시간: {estimated_remaining/60:.1f}분 (완료 예상: {estimated_completion.strftime('%H:%M:%S')})")
        
        pipeline_status["progress"] = progress_percent
        update_status()
        
        if run_script(script, start, end, args.debug):
            success_count += 1
        else:
            failed_scripts.append(script)
    
    # 최종 결과 출력
    total_elapsed_time = time.time() - total_start_time
    logger.info("\n" + "="*50)
    logger.info("📊 파이프라인 실행 결과")
    logger.info(f"• 전체 스크립트: {len(scripts)}개")
    logger.info(f"• 성공: {success_count}개")
    logger.info(f"• 실패: {len(scripts) - success_count}개")
    if failed_scripts:
        logger.info("• 실패한 스크립트:")
        for script in failed_scripts:
            logger.info(f"  - {script}")
    logger.info(f"• 총 소요 시간: {total_elapsed_time/60:.1f}분")
    logger.info("="*50)
    
    # 최종 상태 업데이트
    pipeline_status["progress"] = 100 if success_count == len(scripts) else (success_count / len(scripts) * 100)
    pipeline_status["status"] = "completed" if success_count == len(scripts) else "completed_with_errors"
    pipeline_status["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_status["total_elapsed_minutes"] = round(total_elapsed_time / 60, 1)
    update_status()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️ 사용자에 의해 프로그램이 중단되었습니다.")
        pipeline_status["status"] = "interrupted"
        update_status()
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 예기치 않은 오류 발생: {str(e)}")
        if '--debug' in sys.argv:
            logger.exception("상세 오류:")
        pipeline_status["status"] = "failed"
        pipeline_status["error"] = str(e)
        update_status()
        sys.exit(1)

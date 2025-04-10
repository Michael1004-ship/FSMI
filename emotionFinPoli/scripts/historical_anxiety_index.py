from datetime import datetime, timedelta
# historical_anxiety_index.py
import os
import sys
import logging
import argparse
import subprocess

from pathlib import Path
from google.cloud import storage

# 로그 디렉토리 설정
LOG_ROOT = "/home/hwangjeongmun691/logs"
today = datetime.utcnow().strftime("%Y-%m-%d")
LOG_DATE_DIR = f"{LOG_ROOT}/{today}"

# 디렉토리 생성
os.makedirs(LOG_DATE_DIR, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DATE_DIR}/historical_anxiety_index.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("historical_anxiety")

# 스크립트 경로 (절대 경로 사용)
PROJECT_ROOT = "/home/hwangjeongmun691/projects/emotionFinPoli"
GDELT_FINBERT = os.path.join(PROJECT_ROOT, "scripts/gdelt_realtime_crawling&FinBERT.py")
REDDIT_FINBERT = os.path.join(PROJECT_ROOT, "scripts/reddit_FinBERT.py")
REDDIT_ROBERTA = os.path.join(PROJECT_ROOT, "scripts/reddit_RoBERTa.py")
BUILD_INDEX = os.path.join(PROJECT_ROOT, "building_index/build_anxiety_index.py")

def run_script(script_path, description, env_vars=None):
    """스크립트를 실행하고 결과를 로깅합니다."""
    try:
        logger.info(f"✨ {description} 시작...")
        
        # 환경 변수 설정
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            
        result = subprocess.run(
            [sys.executable, script_path], 
            env=env,
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"[{description}] {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"[{description}] {line}")
        
        logger.info(f"✅ {description} 완료")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 실패: 종료 코드 {e.returncode}")
        if e.stdout:
            logger.error(f"출력: {e.stdout}")
        if e.stderr:
            logger.error(f"에러: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ {description} 실패: {e}")
        return False

def create_modified_script(original_path, output_path, target_date):
    """스크립트 파일을 복사하고 TARGET_DATE 변수를 수정합니다."""
    with open(original_path, 'r') as f:
        content = f.read()
    
    # TARGET_DATE 변수 변경
    if "TARGET_DATE" in content:
        modified_content = content.replace(
            'TARGET_DATE = datetime.today().strftime("%Y-%m-%d")',
            f'TARGET_DATE = "{target_date}"'
        ).replace(
            'TARGET_DATE = datetime.utcnow().strftime("%Y-%m-%d")',
            f'TARGET_DATE = "{target_date}"'
        ).replace(
            'TARGET_DATE = "2024-04-07"',  # 하드코딩된 날짜 변경
            f'TARGET_DATE = "{target_date}"'
        ).replace(
            'TARGET_DATE = "2025-04-07"',  # 하드코딩된 날짜 변경
            f'TARGET_DATE = "{target_date}"'
        )
    else:
        # TARGET_DATE 변수가 없는 경우, 스크립트 상단에 추가
        modified_content = f'TARGET_DATE = "{target_date}"\n' + content
    
    # 수정된 스크립트 저장
    with open(output_path, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"스크립트 수정 완료: {output_path} (TARGET_DATE={target_date})")

def create_modified_gdelt_script(original_path, output_path, target_date):
    """GDELT 스크립트를 수정하여 특정 날짜의 데이터만 처리하도록 합니다."""
    with open(original_path, 'r') as f:
        content = f.read()
    
    # 원본 코드에서 json 파일 찾는 부분을 수정
    modified_content = content.replace(
        'prefix_path = f"{GDELT_PREFIX}{TARGET_DATE}/"',
        f'prefix_path = "news/gdelt/full/"'
    )
    
    # 데이터 처리 코드를 추가하여 특정 날짜 필터링
    filter_code = f'''
# 특정 날짜 데이터 필터링
TARGET_DATE = "{target_date}"
filtered_items = []
for item in items:
    if "DATE" in item:
        item_date = item["DATE"][:8]  # YYYYMMDDHHMMSS 형식에서 YYYYMMDD 추출
        formatted_date = f"{{item_date[:4]}}-{{item_date[4:6]}}-{{item_date[6:8]}}"
        if formatted_date == TARGET_DATE:
            filtered_items.append(item)
items = filtered_items
    '''
    
    # 위 코드를 items 로드 후 위치에 삽입
    modified_content = modified_content.replace(
        'for item in items:',
        f'{filter_code}\nfor item in items:'
    )
    
    # 수정된 스크립트 저장
    with open(output_path, 'w') as f:
        f.write(modified_content)
    
    logger.info(f"GDELT 스크립트 수정 완료: {output_path} (TARGET_DATE={target_date})")

def check_gcs_file_exists(bucket_name, blob_path):
    """GCS에 파일이 존재하는지 확인합니다."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception as e:
        logger.error(f"GCS 파일 확인 중 오류: {e}")
        return False

def verify_output_files(date_str):
    """특정 날짜의 출력 파일이 모두 생성되었는지 확인합니다."""
    bucket_name = "emotion-index-data"
    
    files_to_check = [
        # GDELT FinBERT 결과
        f"news/{date_str}/news_anxiety_index.csv",
        
        # Reddit 분석 결과
        f"reddit/{date_str}/reddit_anxiety_index.csv",
        f"reddit/{date_str}/reddit_anxiety_roberta.csv",
        
        # 최종 불안 지수
        f"final_anxiety_index/{date_str}/anxiety_index_final.csv"
    ]
    
    results = {}
    for file_path in files_to_check:
        exists = check_gcs_file_exists(bucket_name, file_path)
        results[file_path] = exists
        status = "✅ 존재함" if exists else "❌ 존재하지 않음"
        logger.info(f"GCS 파일 확인: gs://{bucket_name}/{file_path} - {status}")
    
    return all(results.values())

def process_date(target_date):
    """특정 날짜의 데이터 처리 및 불안 지수 계산."""
    logger.info(f"🔄 {target_date} 날짜 데이터 처리 시작")
    process_start_time = datetime.now()
    
    # 임시 수정 스크립트 경로
    temp_dir = Path("/tmp/emotion_scripts")
    temp_dir.mkdir(exist_ok=True)
    
    gdelt_temp = temp_dir / "gdelt_finbert_temp.py"
    reddit_finbert_temp = temp_dir / "reddit_finbert_temp.py"
    reddit_roberta_temp = temp_dir / "reddit_roberta_temp.py"
    
    # 스크립트 수정 시작
    logger.info(f"📝 {target_date}용 분석 스크립트 준비 중...")
    script_mod_start = datetime.now()
    
    # GDELT 스크립트는 전체 기간 데이터에서 특정 날짜 필터링 로직 추가
    create_modified_gdelt_script(GDELT_FINBERT, gdelt_temp, target_date)
    
    # 나머지 Reddit 스크립트는 기존 방식대로 수정
    create_modified_script(REDDIT_FINBERT, reddit_finbert_temp, target_date)
    create_modified_script(REDDIT_ROBERTA, reddit_roberta_temp, target_date)
    
    script_mod_time = (datetime.now() - script_mod_start).total_seconds()
    logger.info(f"✅ 스크립트 준비 완료 (소요 시간: {script_mod_time:.2f}초)")
    
    # 단계별 성공 여부 추적
    steps_results = {
        "gdelt": False,
        "reddit_finbert": False,
        "reddit_roberta": False,
        "build_index": False
    }
    
    # 1. GDELT 데이터 분석 - 1/4 단계
    logger.info(f"[1/4] 🔍 {target_date} GDELT 데이터 분석 시작...")
    step_start_time = datetime.now()
    
    gdelt_success = run_script(
        str(gdelt_temp),
        f"GDELT {target_date} 분석"
    )
    
    step_time = (datetime.now() - step_start_time).total_seconds()
    steps_results["gdelt"] = gdelt_success
    
    if not gdelt_success:
        logger.error(f"❌ GDELT {target_date} 분석 실패 (소요 시간: {step_time/60:.2f}분), 계속 진행합니다.")
    else:
        # GCS 저장 확인
        gdelt_saved = check_gcs_file_exists("emotion-index-data", f"news/{target_date}/news_anxiety_index.csv")
        if gdelt_saved:
            logger.info(f"✅ GDELT 분석 완료 (소요 시간: {step_time/60:.2f}분)")
            logger.info(f"  → 저장 경로: gs://emotion-index-data/news/{target_date}/news_anxiety_index.csv")
        else:
            logger.warning(f"⚠️ GDELT 분석은 완료되었으나 결과 파일을 찾을 수 없습니다. (소요 시간: {step_time/60:.2f}분)")
    
    # 2. Reddit FinBERT 분석 - 2/4 단계
    logger.info(f"[2/4] 🔍 {target_date} Reddit FinBERT 분석 시작...")
    step_start_time = datetime.now()
    
    reddit_finbert_success = run_script(
        str(reddit_finbert_temp), 
        f"Reddit FinBERT {target_date} 분석"
    )
    
    step_time = (datetime.now() - step_start_time).total_seconds()
    steps_results["reddit_finbert"] = reddit_finbert_success
    
    if not reddit_finbert_success:
        logger.error(f"❌ Reddit FinBERT {target_date} 분석 실패 (소요 시간: {step_time/60:.2f}분), 계속 진행합니다.")
    else:
        # GCS 저장 확인
        finbert_saved = check_gcs_file_exists("emotion-index-data", f"reddit/{target_date}/reddit_anxiety_index.csv")
        if finbert_saved:
            logger.info(f"✅ Reddit FinBERT 분석 완료 (소요 시간: {step_time/60:.2f}분)")
            logger.info(f"  → 저장 경로: gs://emotion-index-data/reddit/{target_date}/reddit_anxiety_index.csv")
        else:
            logger.warning(f"⚠️ Reddit FinBERT 분석은 완료되었으나 결과 파일을 찾을 수 없습니다. (소요 시간: {step_time/60:.2f}분)")
    
    # 3. Reddit RoBERTa 분석 - 3/4 단계
    logger.info(f"[3/4] 🔍 {target_date} Reddit RoBERTa 분석 시작...")
    step_start_time = datetime.now()
    
    reddit_roberta_success = run_script(
        str(reddit_roberta_temp),
        f"Reddit RoBERTa {target_date} 분석"
    )
    
    step_time = (datetime.now() - step_start_time).total_seconds()
    steps_results["reddit_roberta"] = reddit_roberta_success
    
    if not reddit_roberta_success:
        logger.error(f"❌ Reddit RoBERTa {target_date} 분석 실패 (소요 시간: {step_time/60:.2f}분), 계속 진행합니다.")
    else:
        # GCS 저장 확인
        roberta_saved = check_gcs_file_exists("emotion-index-data", f"reddit/{target_date}/reddit_anxiety_roberta.csv")
        if roberta_saved:
            logger.info(f"✅ Reddit RoBERTa 분석 완료 (소요 시간: {step_time/60:.2f}분)")
            logger.info(f"  → 저장 경로: gs://emotion-index-data/reddit/{target_date}/reddit_anxiety_roberta.csv")
        else:
            logger.warning(f"⚠️ Reddit RoBERTa 분석은 완료되었으나 결과 파일을 찾을 수 없습니다. (소요 시간: {step_time/60:.2f}분)")
    
    # 4. 불안 지수 계산 - 4/4 단계
    logger.info(f"[4/4] 📊 {target_date} 최종 불안 지수 계산 시작...")
    step_start_time = datetime.now()
    
    build_success = run_script(
        BUILD_INDEX,
        f"불안 지수 계산 {target_date}",
        {"PYTHONPATH": PROJECT_ROOT}
    )
    
    step_time = (datetime.now() - step_start_time).total_seconds()
    steps_results["build_index"] = build_success
    
    if not build_success:
        logger.error(f"❌ 불안 지수 계산 {target_date} 실패 (소요 시간: {step_time/60:.2f}분)")
    else:
        # GCS 저장 확인
        final_saved = check_gcs_file_exists("emotion-index-data", f"final_anxiety_index/{target_date}/anxiety_index_final.csv")
        if final_saved:
            logger.info(f"✅ 최종 불안 지수 계산 완료 (소요 시간: {step_time/60:.2f}분)")
            logger.info(f"  → 저장 경로: gs://emotion-index-data/final_anxiety_index/{target_date}/anxiety_index_final.csv")
        else:
            logger.warning(f"⚠️ 불안 지수 계산은 완료되었으나 결과 파일을 찾을 수 없습니다. (소요 시간: {step_time/60:.2f}분)")
    
    # 모든 출력 파일 검증
    logger.info(f"===== {target_date} 처리 결과 요약 =====")
    logger.info("📊 처리 단계별 상태:")
    logger.info(f"  • GDELT 분석: {'✅ 성공' if steps_results['gdelt'] else '❌ 실패'}")
    logger.info(f"  • Reddit FinBERT: {'✅ 성공' if steps_results['reddit_finbert'] else '❌ 실패'}")
    logger.info(f"  • Reddit RoBERTa: {'✅ 성공' if steps_results['reddit_roberta'] else '❌ 실패'}")
    logger.info(f"  • 최종 불안 지수: {'✅ 성공' if steps_results['build_index'] else '❌ 실패'}")
    
    logger.info(f"📁 출력 파일 확인 중...")
    all_files_exist = verify_output_files(target_date)
    
    # 전체 소요 시간 계산
    total_time = (datetime.now() - process_start_time).total_seconds()
    
    if all_files_exist:
        logger.info(f"🎉 {target_date} 모든 출력 파일이 GCS에 성공적으로 저장되었습니다.")
        logger.info(f"⏱️ 총 소요 시간: {total_time/60:.2f}분")
    else:
        logger.warning(f"⚠️ {target_date} 일부 출력 파일이 GCS에 저장되지 않았습니다.")
        logger.warning(f"⏱️ 총 소요 시간: {total_time/60:.2f}분")
    
    success_rate = sum(1 for v in steps_results.values() if v) / len(steps_results) * 100
    logger.info(f"🔢 성공률: {success_rate:.1f}% ({sum(1 for v in steps_results.values() if v)}/{len(steps_results)})")
    
    return build_success

def process_date_range(start_date, end_date):
    """날짜 범위에 대한 데이터 처리."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 총 일수 계산
    total_days = (end_dt - start_dt).days + 1
    logger.info(f"🗓️ 총 {total_days}일 데이터 처리 시작: {start_date} ~ {end_date}")
    
    # 전체 실행 시작 시간
    range_start_time = datetime.now()
    
    # 결과 저장
    results = []
    successful_days = 0
    
    # 시작일부터 종료일까지 반복
    current_dt = start_dt
    for day_index in range(total_days):
        date_str = current_dt.strftime("%Y-%m-%d")
        
        # 진행 상황 표시
        progress = (day_index / total_days) * 100
        elapsed = (datetime.now() - range_start_time).total_seconds()
        remaining = 0 if day_index == 0 else (elapsed / day_index) * (total_days - day_index)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 전체 진행률: {progress:.1f}% - {day_index+1}/{total_days}일 처리 중")
        logger.info(f"⏱️ 예상 남은 시간: {remaining/60:.1f}분 ({remaining/3600:.2f}시간)")
        logger.info(f"📅 현재 처리 날짜: {date_str} ({day_index+1}일차)")
        logger.info(f"{'='*50}\n")
        
        # 해당 날짜 처리
        day_start_time = datetime.now()
        success = process_date(date_str)
        day_elapsed = (datetime.now() - day_start_time).total_seconds()
        
        # 결과 저장
        status = "✅ 성공" if success else "❌ 실패"
        results.append((date_str, success, day_elapsed))
        
        if success:
            successful_days += 1
            
        logger.info(f"📋 {date_str} 처리 결과: {status} (소요 시간: {day_elapsed/60:.2f}분)")
        logger.info(f"현재까지 {successful_days}/{day_index+1} 일 성공 (성공률: {(successful_days/(day_index+1))*100:.1f}%)")
        
        # 다음 날짜로 이동
        current_dt += timedelta(days=1)
    
    # 전체 실행 결과 요약
    total_elapsed = (datetime.now() - range_start_time).total_seconds()
    success_rate = (successful_days / total_days) * 100
    
    logger.info("\n" + "="*60)
    logger.info(f"📊 {start_date} ~ {end_date} 처리 최종 결과")
    logger.info(f"  • 총 처리 일수: {total_days}일")
    logger.info(f"  • 성공한 일수: {successful_days}일")
    logger.info(f"  • 실패한 일수: {total_days - successful_days}일")
    logger.info(f"  • 성공률: {success_rate:.1f}%")
    logger.info(f"  • 총 소요 시간: {total_elapsed/60:.2f}분 ({total_elapsed/3600:.2f}시간)")
    logger.info(f"  • 일 평균 소요 시간: {(total_elapsed/total_days)/60:.2f}분")
    
    # 일별 처리 결과 상세 표시
    logger.info("\n🔍 일별 처리 결과:")
    for date_str, success, elapsed in results:
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"  • {date_str}: {status} (소요 시간: {elapsed/60:.2f}분)")
    
    logger.info("="*60)
    logger.info(f"🎉 {start_date} ~ {end_date} 전체 처리 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="과거 데이터로 불안 지수 계산")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()
    
    # 명령줄 인자가 없으면 사용자 입력 요청
    if not args.start or not args.end:
        print("\n===== 과거 데이터 불안 지수 계산 =====")
        print("처리할 날짜 범위를 입력해주세요 (YYYY-MM-DD 형식)")
        if not args.start:
            args.start = input("시작일: ")
        if not args.end:
            args.end = input("종료일: ")
    
    # 날짜 형식 검증
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        
        # 날짜 범위 유효성 검사
        if start_dt > end_dt:
            print("오류: 시작일이 종료일보다 늦습니다.")
            sys.exit(1)
            
        # 너무 긴 범위 경고
        days_diff = (end_dt - start_dt).days
        if days_diff > 30:
            confirm = input(f"⚠️ {days_diff}일의 긴 기간을 처리합니다. 계속하시겠습니까? (y/n): ")
            if confirm.lower() != 'y':
                print("작업이 취소되었습니다.")
                sys.exit(0)
                
    except ValueError:
        print("오류: 날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용하세요.")
        sys.exit(1)
    
    print(f"\n처리 시작: {args.start} ~ {args.end}")
    print(f"총 {(end_dt - start_dt).days + 1}일 데이터 처리 예정")
    print("=" * 40)
    
    logger.info(f"과거 데이터 처리 시작: {args.start} ~ {args.end}")
    process_date_range(args.start, args.end)
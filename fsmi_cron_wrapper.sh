# 파일 생성: /home/hwangjeongmun691/projects/fsmi_cron_wrapper.sh
#!/bin/bash

# 날짜 디렉토리 생성
TODAY=$(date +"%Y-%m-%d")
LOG_DIR="/home/hwangjeongmun691/logs/$TODAY"
TIME=$(date +"%H%M%S")
LOG_FILE="$LOG_DIR/fsmi_cron_${TODAY//-/}_$TIME.log"

# 디렉토리 생성
mkdir -p "$LOG_DIR"

# 시작 시간과 정보 기록
echo "======== FSMI 크론 작업 시작: $(date) ========" > "$LOG_FILE"
echo "실행 환경: $(whoami)@$(hostname)" >> "$LOG_FILE"

# 메인 스크립트 실행
cd /home/hwangjeongmun691/projects && \
/home/hwangjeongmun691/projects/emotionFinPoli/env/bin/python3 \
/home/hwangjeongmun691/projects/FSMI_operator.py >> "$LOG_FILE" 2>&1

# 종료 상태 기록
EXIT_CODE=$?
echo "======== 작업 종료: $(date) | 상태 코드: $EXIT_CODE ========" >> "$LOG_FILE"

# 기존 로그 파일에도 작업 기록 남기기 (나중에 참조용)
echo "$(date): 작업 완료, 로그 위치: $LOG_FILE (상태: $EXIT_CODE)" >> "/home/hwangjeongmun691/logs/cron_fsmi_index.log"

exit $EXIT_CODE

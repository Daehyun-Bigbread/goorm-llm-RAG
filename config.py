# pronun_model/config.py

from dotenv import load_dotenv
from pathlib import Path
import os
from fastapi import HTTPException
import logging

# 모듈별 로거 생성
logger = logging.getLogger(__name__)  # 'server.config' 로거 사용

# .env 파일에서 환경 변수 로드
load_dotenv()

# Docker 환경 감지 및 경로 설정
try:
    with open('/proc/1/cgroup', 'rt') as f:
        cgroup_content = f.read()
    if 'docker' in cgroup_content:
        # Docker 환경
        BASE_DIR = Path("/app")
        logger.info("Docker 환경으로 감지되었습니다.")
        logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
    else:
        # 로컬 환경
        BASE_DIR = Path(__file__).resolve().parent.parent
        logger.info("로컬 환경으로 감지되었습니다.")
        logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
except FileNotFoundError:
    # 로컬 환경
    BASE_DIR = Path(__file__).resolve().parent.parent
    logger.info("로컬 환경으로 간주합니다. 기본 경로로 설정합니다.")
    logger.debug(f"BASE_DIR 설정: {BASE_DIR}")
except Exception as e:
    logger.error(f"환경 감지 중 오류 발생: {e}", extra={
        "errorType": "EnvironmentDetectionError",
        "error_message": str(e)
    })
    raise HTTPException(status_code=500, detail="환경 감지 중 오류 발생") from e
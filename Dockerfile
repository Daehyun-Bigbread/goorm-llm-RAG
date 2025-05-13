# 베이스 이미지로 Python 3.10 슬림 버전 사용
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app

# 환경 변수 파일(.env) 복사
# .env 파일은 빌드 시 포함하지 않도록 .dockerignore 설정을 권장
COPY .env /app/.env

# Uvicorn으로 FastAPI 앱 실행
# uvicorn 실행 커맨드를 ENTRYPOINT나 CMD에 정의
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

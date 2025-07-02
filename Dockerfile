# 베이스 이미지: Python 3.12
FROM python:3.12-slim

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 파일 복사
COPY . /app

# pip 업그레이드 및 의존성 설치
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# chroma_db가 필요하다면 미리 생성하거나 복사 (보통은 runtime에 persist됨)

# 포트 설정
EXPOSE 8000

# 서버 실행 명령어
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

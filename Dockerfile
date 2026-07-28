# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ARG APP_VERSION=dev
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="RoleFit Resume Agent" \
      org.opencontainers.image.description="Dual-engine Google ADK and LangGraph resume application" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN groupadd --system rolefit \
    && useradd --system --gid rolefit --home-dir /app --shell /usr/sbin/nologin rolefit

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=rolefit:rolefit app ./app
COPY --chown=rolefit:rolefit resume_agent ./resume_agent
RUN mkdir -p /app/generated_pdfs \
    && chown rolefit:rolefit /app/generated_pdfs

USER rolefit

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/resume/health', timeout=3)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]

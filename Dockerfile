FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8088
EXPOSE 8088

CMD python apps/site_design_platform/start.py \
    --host 0.0.0.0 \
    --port ${PORT} \
    --no-open

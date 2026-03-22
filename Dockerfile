# 1. Gunakan image Python resmi yang ringan
FROM python:3.9-slim

# 2. Set working directory di dalam kontainer
WORKDIR /app

# 3. Instal dependensi sistem yang dibutuhkan untuk psycopg2 dan XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Salin file requirements.txt terlebih dahulu (untuk caching layer)
COPY requirements.txt .

# 5. Instal library Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Salin semua file project ke dalam kontainer
COPY . .

# 7. Expose port yang akan digunakan FastAPI
EXPOSE 8000

# 8. Jalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

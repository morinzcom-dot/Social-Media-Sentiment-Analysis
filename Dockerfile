FROM python:3.12-slim

WORKDIR /app

# تثبيت المتطلبات أولاً (يستفيد من Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي المشروع
COPY . .

# المنفذ الذي يستمع عليه Uvicorn داخل الحاوية
EXPOSE 8000

# معظم منصات الاستضافة (Render, Railway, Fly.io) تمرر رقم المنفذ
# عبر متغير البيئة PORT — نستخدمه إن وُجد، وإلا نستخدم 8000 افتراضياً
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

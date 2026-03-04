FROM python:3.9-slim

WORKDIR /app

# Kebutuhan library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin semua file dari folder Membangun_model ke dalam kontainer
COPY . .

# Menjalankan model tuning saat kontainer dijalankan
CMD ["python", "modelling_tuning.py"]
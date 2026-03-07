FROM python:3.9-slim

WORKDIR /app

# Menyalin kebutuhan library
COPY MLProject/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek ke dalam kontainer
COPY . .

# Menjalankan model tuning saat kontainer aktif
CMD ["python", "MLProject/modelling_tuning.py"]

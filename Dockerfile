# ใช้ Base Image ของ Python 3.9
FROM python:3.9-slim

# ตั้งค่าที่ทำงานใน container
WORKDIR /app

# ติดตั้งเครื่องมือพัฒนาและ dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# คัดลอกไฟล์ทั้งหมดไปยัง container
COPY . .

# อัปเดต pip และติดตั้ง dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# เปิดพอร์ต
EXPOSE 8000

# รันเซิร์ฟเวอร์เมื่อ container เริ่มต้น
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

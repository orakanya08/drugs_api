from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import shutil
import os
import tempfile
from ultralytics import YOLO
from base64 import b64decode
import uuid

app = FastAPI()

# ตั้งค่า static และ templates 
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# โหลดโมเดล YOLO
model = None

async def load_model():
    global model
    model_path = "api/models/drugs_yolov8.pt"  # แก้ไขเส้นทางโมเดลให้ถูกต้อง
    try:
        model = YOLO(model_path)
        print("Model loaded and running on CPU.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")


def clean_old_results():
    # ตรวจสอบว่าโฟลเดอร์ `runs/detect` มีอยู่หรือไม่
    results_dir = "runs/detect"
    if not os.path.exists(results_dir):
        print(f"Directory '{results_dir}' does not exist. Skipping cleanup.")
        return

    # ลบ subdirectories เก่าใน `runs/detect`
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for subdir in subdirs:
        try:
            shutil.rmtree(subdir)
            print(f"Deleted old results directory: {subdir}")
        except Exception as e:
            print(f"Error deleting directory {subdir}: {e}")

# โฟลเดอร์สำหรับเก็บไฟล์ภาพ (ใช้โฟลเดอร์ชั่วคราว)
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = "api/static/runs/detect"  # จัดเก็บไฟล์ใน static

# ชื่อคลาสและคำแนะนำการใช้งาน run
class_names = {
    0: "alaxan", 1: "bactidol", 2: "bioflu", 3: "biogesic", 4: "dayzinc",
    5: "decolgen", 6: "fishoil", 7: "kremil", 8: "medicol", 9: "neozep"
}

class_usage = {
    "alaxan": "รับประทานครั้งละ 1 เม็ดทุก 4-6 ชั่วโมงหลังอาหาร",
    "bactidol": "กลั้วคอด้วยปริมาณ 10 มิลลิลิตร 2-3 ครั้งต่อวัน",
    "bioflu": "รับประทานครั้งละ 1 เม็ดทุก 6 ชั่วโมง",
    "biogesic": "รับประทานครั้งละ 1-2 เม็ดทุก 4-6 ชั่วโมง",
    "dayzinc": "รับประทานวันละ 1 เม็ดหลังอาหาร",
    "decolgen": "รับประทานครั้งละ 1 เม็ดทุก 6 ชั่วโมง",
    "fishoil": "รับประทานวันละ 1-2 เม็ดพร้อมอาหาร",
    "kremil": "เคี้ยวเม็ดครั้งละ 1-2 เม็ดหลังอาหาร",
    "medicol": "รับประทานครั้งละ 1 เม็ดทุก 4-6 ชั่วโมง",
    "neozep": "รับประทานครั้งละ 1 เม็ดทุก 6 ชั่วโมง"
}

# ฟังก์ชันสำหรับหาโฟลเดอร์ผลลัพธ์ล่าสุด
def get_latest_result_folder(base_dir="runs/detect"):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    return max(subdirs, key=os.path.getmtime) if subdirs else None

@app.on_event("startup")
async def startup_event():
    # สร้างโฟลเดอร์ `runs/detect` ถ้าไม่มี
    os.makedirs("runs/detect", exist_ok=True)
    print("Ensured 'runs/detect' directory exists.")

    # โหลดโมเดล
    await load_model()

    # ทำความสะอาดผลลัพธ์เก่า
    clean_old_results()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # บันทึกไฟล์อัปโหลด
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ใช้โมเดล YOLO ทำนาย
        results = model.predict(source=file_path, save=True)

        # ค้นหาโฟลเดอร์ผลลัพธ์ล่าสุด
        latest_result_folder = get_latest_result_folder()
        if not latest_result_folder:
            return {"error": "No prediction folder found. Please check the YOLO output directory."}

        # สร้างเส้นทางไฟล์ผลลัพธ์
        result_image_path = os.path.join(latest_result_folder, os.path.basename(file.filename))
        if not os.path.exists(result_image_path):
            return {"error": f"File not found: {result_image_path}"}

        # คัดลอกไฟล์ไปยังโฟลเดอร์ static
        dest_path = os.path.join(RESULT_FOLDER, file.filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)  # ตรวจสอบและสร้างโฟลเดอร์ static
        shutil.copy(result_image_path, dest_path)

        # เตรียม URL ของไฟล์ภาพผลลัพธ์
        image_url = dest_path.replace("api/static/", "/static/")

        # เตรียมข้อมูลคำแนะนำการใช้งาน
        predictions = []
        usage_instructions = []
        for result in results:
            for cls in result.boxes.cls.tolist():
                cls_name = class_names[int(cls)]
                predictions.append(cls_name)
                usage_instructions.append((cls_name, class_usage.get(cls_name, "ไม่มีข้อมูล")))

        # ส่งข้อมูลไปที่ template สำหรับแสดงผล
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "image_url": image_url,
                "predictions": predictions,
                "usage_instructions": usage_instructions,
            },
        )
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}



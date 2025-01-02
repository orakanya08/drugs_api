from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import tempfile
from ultralytics import YOLO
from base64 import b64decode
import uuid

# สร้างแอป FastAPI
app = FastAPI()

# ตั้งค่ารูปแบบไฟล์ static และ templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# โหลดโมเดล YOLO
MODEL_PATH = "api/models/drugs_yolov8.pt"
model = YOLO(MODEL_PATH)

# โฟลเดอร์สำหรับเก็บไฟล์ภาพ (ใช้โฟลเดอร์ชั่วคราว)
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = tempfile.mkdtemp()

# ชื่อคลาสและคำแนะนำการใช้งาน
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
        
        result_image_path = os.path.join(latest_result_folder, file.filename)
        dest_path = os.path.join(RESULT_FOLDER, file.filename)
        shutil.copy(result_image_path, dest_path)

        image_url = dest_path.replace("api/static/", "/static/")
        predictions = [
            (class_names[int(cls)], class_usage[class_names[int(cls)]])
            for result in results for cls in result.boxes.cls.tolist()
        ]

        return templates.TemplateResponse(
            "result.html",
            {"request": request, "image_url": image_url, "predictions": predictions},
        )
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

@app.on_event("startup")
def clean_old_results():
    subdirs = [os.path.join("runs/detect", d) for d in os.listdir("runs/detect") if os.path.isdir(os.path.join("runs/detect", d))]
    for folder in subdirs[:-1]:  # เก็บโฟลเดอร์ล่าสุดไว้
        shutil.rmtree(folder)

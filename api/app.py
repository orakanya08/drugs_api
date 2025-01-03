from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import os
from ultralytics import YOLO

app = FastAPI()

# ตั้งค่า static และ templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="templates")

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


@app.on_event("startup")
async def startup_event():
    # สร้างโฟลเดอร์ `runs/detect` ถ้าไม่มี
    os.makedirs("runs/detect", exist_ok=True)
    print("Ensured 'runs/detect' directory exists.")

    # โหลดโมเดล
    await load_model()

    # ทำความสะอาดผลลัพธ์เก่า
    clean_old_results()


@app.post("/upload/")
async def upload_file(file: UploadFile):
    try:
        # บันทึกไฟล์ที่อัพโหลด
        file_location = f"runs/detect/{file.filename}"
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # รันการตรวจจับ
        results = model(file_location)
        results_dir = results.save_dir

        return {"message": "File uploaded successfully", "results_dir": str(results_dir)}
    except Exception as e:
        return {"error": f"Error processing file: {e}"}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

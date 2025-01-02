# ใช้ Python 3.9 เป็น base image
FROM python:3.9-slim

# ตั้งค่าที่ทำงานใน container
WORKDIR /app

# ติดตั้งเครื่องมือพัฒนา
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    && apt-get clean

# คัดลอกไฟล์ทั้งหมดจากโฟลเดอร์ปัจจุบันไปยัง container
COPY . .

# อัปเดต pip และติดตั้งแพ็กเกจที่ต้องการ RUN pip install asttokens==2.4.1
RUN pip install --upgrade pip
RUN pip install torch==1.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install absl-py==2.1.0
RUN pip install annotated-types==0.7.0
RUN pip install anyio==4.6.2.post1
RUN pip install asttokens==2.4.1
RUN pip install astunparse==1.6.3
RUN pip install certifi==2024.8.30
RUN pip install charset-normalizer==3.4.0
RUN pip install click==8.1.7
RUN pip install cmake==3.31.1
RUN pip install colorama==0.4.6
RUN pip install comm==0.2.2
RUN pip install contourpy==1.2.1
RUN pip install cycler==0.12.1
RUN pip install debugpy==1.8.9
RUN pip install decorator==5.1.1
RUN pip install dlib==19.24.6
RUN pip install exceptiongroup==1.2.2
RUN pip install executing==2.1.0
RUN pip install fastapi==0.115.5
RUN pip install filelock==3.16.1
RUN pip install flatbuffers==24.3.25
RUN pip install fonttools==4.55.0
RUN pip install fsspec==2024.10.0
RUN pip install gast==0.6.0
RUN pip install gitdb==4.0.11
RUN pip install GitPython==3.1.43
RUN pip install google-pasta==0.2.0
RUN pip install grpcio==1.68.0
RUN pip install h11==0.14.0
RUN pip install h5py==3.12.1
RUN pip install idna==3.10
RUN pip install ipykernel==6.29.5
RUN pip install ipython==8.18.0
RUN pip install jedi==0.19.2
RUN pip install Jinja2==3.1.4
RUN pip install joblib==1.4.2
RUN pip install jupyter_client==8.6.3
RUN pip install jupyter_core==5.7.2
RUN pip install keras==3.6.0
RUN pip install kiwisolver==1.4.7
RUN pip install libclang==18.1.1
RUN pip install Markdown==3.7
RUN pip install markdown-it-py==3.0.0
RUN pip install MarkupSafe==3.0.2
RUN pip install matplotlib==3.9.3
RUN pip install matplotlib-inline==0.1.7
RUN pip install mdurl==0.1.2
RUN pip install ml-dtypes==0.4.1
RUN pip install mpmath==1.3.0
RUN pip install namex==0.0.8
RUN pip install nest-asyncio==1.6.0
RUN pip install networkx==3.2.1
RUN pip install numpy==1.24.4
RUN pip install opencv-python==4.7.0.72
RUN pip install opencv-python-headless==4.7.0.72
RUN pip install opt_einsum==3.3.0
RUN pip install optree==0.5.0
RUN pip install packaging==21.3
RUN pip install pandas==1.5.3
RUN pip install parso==0.8.3
RUN pip install pillow==9.5.0
RUN pip install platformdirs==3.10.0
RUN pip install prompt_toolkit==3.0.38
RUN pip install protobuf==4.23.3
RUN pip install psutil==5.9.5
RUN pip install pure_eval==0.2.2
RUN pip install py-cpuinfo==9.0.0
RUN pip install pydantic==1.10.13
RUN pip install pydantic_core==2.3.1
RUN pip install Pygments==2.15.1
RUN pip install pyparsing==3.0.9
RUN pip install python-dateutil==2.8.2
RUN pip install python-multipart==0.0.5
RUN pip install pytz==2023.3
RUN pip install PyYAML==6.0.1
RUN pip install pyzmq==24.0.1
RUN pip install requests==2.31.0
RUN pip install rich==13.5.2
RUN pip install scikit-learn==1.2.2
RUN pip install scipy==1.10.1
RUN pip install seaborn==0.12.2
RUN pip install six==1.16.0
RUN pip install smmap==5.0.0
RUN pip install sniffio==1.3.0
RUN pip install stack-data==0.6.2
RUN pip install starlette==0.27.0
RUN pip install sympy==1.12
RUN pip install tensorboard==2.12.3
RUN pip install tensorboard-data-server==0.7.0
RUN pip install tensorflow==2.12.0
RUN pip install tensorflow-io-gcs-filesystem==0.30.0
RUN pip install termcolor==2.4.0
RUN pip install thop==0.1.1.post2209072238
RUN pip install threadpoolctl==3.1.0
RUN pip install torch==2.0.1
RUN pip install torchvision==0.15.2
RUN pip install tornado==6.3.2
RUN pip install tqdm==4.65.0
RUN pip install traitlets==5.9.0
RUN pip install typing_extensions==4.5.0
RUN pip install tzdata==2023.3
RUN pip install ultralytics==8.0.43
RUN pip install ultralytics-thop
RUN pip install urllib3==1.26.16
RUN pip install uvicorn==0.22.0
RUN pip install wcwidth==0.2.8
RUN pip install Werkzeug==2.3.7
RUN pip install wrapt==1.14.1

# รันเซิร์ฟเวอร์เมื่อ container เริ่มต้น
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

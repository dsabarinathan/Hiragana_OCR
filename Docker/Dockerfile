FROM tensorflow/tensorflow:2.8.0

WORKDIR /app

# Install OpenCV package
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Install OpenCV
RUN pip install opencv-python-headless
#Install Libraries
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip install flask
RUN pip install requests
RUN pip install numpy
RUN pip install tensorflow-addons
RUN pip install scikit-image

COPY ./models/model-ocr-0.1829.h5 /app/models/model-ocr-0.1829.h5

COPY config.json /app/config.json

COPY get_coordinate.py /app/get_coordinate.py

COPY hiragana_ocr_engine.py /app/hiragana_ocr_engine.py

CMD ["python", "/app/hiragana_ocr_engine.py"]


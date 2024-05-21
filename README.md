# YOLOv8 PPE Compliance Detection Website Using FastAPI

## Overview

This project implements a web application for Personal Protective Equipment (PPE) compliance detection using YOLOv8. The application allows users to upload images and receive predictions on PPE compliance.

## Project Structure

```
hosting/
│
├── main.py
├── yolofastapi/
│   ├── routers/
│   ├── schemas/
│   ├── detectors/
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── weights/
│   └── best.pt
└── README.md
```

### `main.py`
The entry point of the application. Sets up the FastAPI server and includes the YOLO router.

### `yolofastapi/routers`
Defines the API endpoints for image upload and retrieval.

### `yolofastapi/schemas`
Defines the request and response schemas using Pydantic.

### `yolofastapi/detectors`
Contains the class for running the YOLOv8 model and processing images.

### `static/`
Contains the frontend files for the web interface.

### `weights/`
Contains the trained YOLOv8 model weights.

## Features

- **Upload Images**: Users can upload images for PPE compliance detection.
- **YOLOv8 Detection**: The application uses a custom-trained YOLOv8 model to detect PPE compliance.
- **Retrieve Analyzed Images**: Users can retrieve analyzed images with detected labels.

### Example Website outputs:

![alt text](https://github.com/DominicOcharo/YOLOv8-With-FastAPI/blob/main/images/webresults2.png?raw=true)

![alt text](https://github.com/DominicOcharo/YOLOv8-With-FastAPI/blob/main/images/webresults2.png?raw=true)

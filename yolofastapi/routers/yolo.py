from fastapi import APIRouter, UploadFile, Response, status, HTTPException, Depends
from sqlalchemy.orm import Session
import cv2
import numpy as np
from yolofastapi.detectors import yolov8
from yolofastapi.schemas.yolo import ImageAnalysisResponse, FilteredImageAnalysisResponse
from yolofastapi.models import ImageAnalysis
# from yolofastapi import get_db

router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")

images = []

@router.post("",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image."}
    },
    response_model=ImageAnalysisResponse,
)
async def yolo_image_upload(file: UploadFile) -> ImageAnalysisResponse:
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels_confidences = await dt()

    labels, confidences = zip(*labels_confidences) if labels_confidences else ([], [])
    
    success, encoded_image = cv2.imencode(".png", frame)
    images.append((encoded_image, labels_confidences))

    # db_image_analysis = ImageAnalysis(
    #     labels=",".join(labels),
    #     confidences=",".join(map(str, confidences))
    # )
    # db.add(db_image_analysis)
    # db.commit()
    # db.refresh(db_image_analysis)

    return ImageAnalysisResponse(
        id=0,
        labels=list(labels), 
        confidences=list(confidences)
    )

@router.post("/filtered",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image and Filtered Results."}
    },
    response_model=FilteredImageAnalysisResponse,
)
async def yolo_image_upload_filtered(file: UploadFile) -> FilteredImageAnalysisResponse:
    contents = await file.read()
    
    # Convert the uploaded file contents to a NumPy array
    np_array = np.frombuffer(contents, np.uint8)
    
    # Decode the NumPy array to an image
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Check if the image has 4 channels (RGBA)
    if image.shape[2] == 4:
        # Convert from RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Convert the image back to bytes after processing (optional)
    _, encoded_image = cv2.imencode('.png', image)
    processed_contents = encoded_image.tobytes()
    
    # Perform YOLO object detection on the image
    dt = yolov8.YoloV8ImageObjectDetection(chunked=processed_contents)
    frame, labels_confidences = await dt()
    
    filtered_labels = []
    filtered_confidences = []
    for label, confidence in labels_confidences:
        if label in ["Hardhat", "Person", "Safety Vest"]:
            filtered_labels.append(label)
            filtered_confidences.append(confidence)

    person_count = filtered_labels.count("Person")
    hardhat_count = filtered_labels.count("Hardhat")
    safety_vest_count = filtered_labels.count("Safety Vest")

    percentage = 0.0
    recommendation = ""

    if person_count == 0:
        recommendation = "Invalid, NA"
    else:
        expected_count = person_count
        hardhat_percentage = (hardhat_count / expected_count) * 100
        safety_vest_percentage = (safety_vest_count / expected_count) * 100
        overall_percentage = (hardhat_percentage + safety_vest_percentage) / 2
        percentage = overall_percentage

        if overall_percentage >= 90:
            recommendation = "Approve"
        elif 70 <= overall_percentage < 90:
            recommendation = "Inspect"
        elif 50 <= overall_percentage < 70:
            recommendation = "Reject"
        else:
            recommendation = "Reject"

    success, encoded_image = cv2.imencode(".png", frame)
    images.append((encoded_image, labels_confidences))

    # db_image_analysis = ImageAnalysis(
    #     filtered_labels=",".join(filtered_labels),
    #     filtered_confidences=",".join(map(str, filtered_confidences)),
    #     recommendation=recommendation,
    #     percentage=percentage
    # )
    # db.add(db_image_analysis)
    # db.commit()
    # db.refresh(db_image_analysis)

    return FilteredImageAnalysisResponse(
        id=0,
        filtered_labels=filtered_labels,
        filtered_confidences=filtered_confidences,
        recommendation=recommendation,
        percentage=percentage
    )

@router.get(
    "/{image_id}",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"content": {"image/png": {}}},
        404: {"description": "Image ID Not Found."}
    },
    response_class=Response,
)
async def yolo_image_download(image_id: int) -> Response:
    # db_image = db.query(ImageAnalysis).filter(ImageAnalysis.id == image_id).first()
    # if db_image is None:
    #     raise HTTPException(status_code=404, detail="Image not found")
    
    encoded_image, labels_confidences = images[image_id - 1]
    labels, confidences = zip(*labels_confidences) if labels_confidences else ([], [])
    headers = {
        "labels": ",".join(labels),
        "confidences": ",".join(map(str, confidences))
    }
    return Response(content=encoded_image.tobytes(), media_type="image/png", headers=headers)

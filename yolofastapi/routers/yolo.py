from fastapi import APIRouter, UploadFile, Response, status, HTTPException
from yolofastapi.detectors import yolov8
import cv2
from yolofastapi.schemas.yolo import ImageAnalysisResponse, FilteredImageAnalysisResponse

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
    """Takes a multi-part upload image and runs yolov8 on it to detect objects

    Arguments:
        file (UploadFile): The multi-part upload file

    Returns:
        response (ImageAnalysisResponse): The image ID, labels, and confidences 
                                          in the pydantic object
    """
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels_confidences = await dt()
    
    labels, confidences = zip(*labels_confidences) if labels_confidences else ([], [])

    success, encoded_image = cv2.imencode(".png", frame)
    images.append((encoded_image, labels_confidences))
    return ImageAnalysisResponse(
        id=len(images), 
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
    """Takes a multi-part upload image and runs yolov8 on it to detect objects
    Filters results for Hardhat, Person, and Safety Vest.

    Arguments:
        file (UploadFile): The multi-part upload file

    Returns:
        response (FilteredImageAnalysisResponse): The image ID, filtered labels, 
                                                  and filtered confidences in the 
                                                  pydantic object
    """
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels_confidences = await dt()
    
    filtered_labels = []
    filtered_confidences = []
    for label, confidence in labels_confidences:
        if label in ["Hardhat", "Person", "Safety Vest"]:
            filtered_labels.append(label)
            filtered_confidences.append(confidence)

    success, encoded_image = cv2.imencode(".png", frame)
    images.append((encoded_image, labels_confidences))
    return FilteredImageAnalysisResponse(
        id=len(images),
        filtered_labels=filtered_labels,
        filtered_confidences=filtered_confidences
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
    """Takes an image id as a path param and returns that encoded
    image from the images array along with the detected labels.

    Arguments:
        image_id (int): The image ID to download

    Returns:
        response (Response): The encoded image in PNG format along with the labels
    """
    try:
        encoded_image, labels_confidences = images[image_id - 1]
        labels, confidences = zip(*labels_confidences) if labels_confidences else ([], [])
        headers = {
            "labels": ",".join(labels),
            "confidences": ",".join(map(str, confidences))
        }
        return Response(content=encoded_image.tobytes(), media_type="image/png", headers=headers)
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found")

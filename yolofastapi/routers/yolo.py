# For API operations and standards
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
# Our detector objects
from yolofastapi.detectors import yolov8
# For encoding images
import cv2
# For response schemas
from yolofastapi.schemas.yolo import ImageAnalysisResponse

# A new router object that we can add endpoints to.
# Note that the prefix is /yolo, so all endpoints from
# here on will be relative to /yolo
router = APIRouter(tags=["Image Upload and analysis"], prefix="/yolo")

# A cache of annotated images. Note that this would typically
# be some sort of persistent storage (think maybe postgres + S3)
# but for simplicity, we can keep things in memory
images = []

@router.post("/",
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
        response (ImageAnalysisResponse): The image ID and labels in 
                                          the pydantic object
    """
    contents = await file.read()
    dt = yolov8.YoloV8ImageObjectDetection(chunked=contents)
    frame, labels = await dt()
    success, encoded_image = cv2.imencode(".png", frame)
    images.append((encoded_image, labels))
    return ImageAnalysisResponse(id=len(images), labels=labels)


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
        encoded_image, labels = images[image_id - 1]
        return Response(content=encoded_image.tobytes(), media_type="image/png", headers={"labels": ",".join(labels)})
    except IndexError:
        raise HTTPException(status_code=404, detail="Image not found")

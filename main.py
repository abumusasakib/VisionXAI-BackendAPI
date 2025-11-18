import os
import logging
import socket
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ImgCap import captioner as cap
from loguru import logger
from zeroconf import ServiceInfo, Zeroconf
from contextlib import asynccontextmanager

# Configure logging with loguru
LOG_FILE = "logs/app.log"

logger.add(
    LOG_FILE,
    rotation="1 day",  # Rotate log every day
    retention="7 days",  # Keep logs for the last 7 days
    compression="zip",  # Compress old log files
    level="INFO",  # Default log level
)

# Redirect FastAPI/uvicorn logs to loguru
logging.getLogger("uvicorn.access").handlers = [logging.StreamHandler()]
logging.getLogger("uvicorn.error").handlers = [logging.StreamHandler()]

# Initialize the FastAPI app
app = FastAPI(
    title="Image Caption Generation API",
    description="An API for generating image captions in Bengali.",
    version="1.0.0",
)

# Configure upload folder
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info(f"Upload folder initialized at {UPLOAD_FOLDER}")

# Allow CORS for flexibility in development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware configured")

# mDNS Configuration
zeroconf = Zeroconf()
service_info = None


# mDNS Registration
def register_mdns_service():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 5353
    service_type = "_http._tcp.local."
    service_name = "VisionXAI API._http._tcp.local."
    service_properties = {"description": "Image Captioning API in Bengali"}

    global service_info
    service_info = ServiceInfo(
        service_type,
        service_name,
        addresses=[socket.inet_aton(local_ip)],
        port=port,
        properties=service_properties,
        server=f"{hostname}.local.",
    )
    logger.info(f"Registering service: {service_info}")

    zeroconf = Zeroconf()
    logger.info("Attempting to register mDNS service...")
    try:
        zeroconf.register_service(service_info)
        logger.info("mDNS service registered successfully.")
    except Exception as e:
        logger.error(f"Error registering mDNS service: {e}")

    logger.info(f"mDNS service registered: {service_info.name} on {local_ip}:5353")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global zeroconf, service_info
    if service_info:
        logger.info("Unregistering mDNS service...")
        zeroconf.unregister_service(service_info)
        zeroconf.close()
        logger.info("mDNS service unregistered.")


app = FastAPI(lifespan=lifespan)


# Routes and Endpoints


@app.get("/", response_model=dict)
def read_root():
    """
    Root endpoint to confirm the API is live.
    """
    logger.info("Root endpoint accessed")
    return {"message": "Image Caption Generation in Bengali"}


@app.post("/upload", response_model=dict)
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to upload an image. Replaces any existing image in the upload folder.
    """
    logger.info("Upload endpoint called")
    if not image.filename:
        logger.warning("No file selected by the user")
        raise HTTPException(status_code=400, detail="No file selected.")

    extension = image.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        logger.warning(f"Invalid file format: {extension}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only jpg, jpeg, png are supported.",
        )

    # Clear the upload folder
    logger.info("Clearing existing files in the upload folder")
    for item in os.listdir(UPLOAD_FOLDER):
        item_path = os.path.join(UPLOAD_FOLDER, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

    # Save the new image
    file_path = os.path.join(UPLOAD_FOLDER, f"image.{extension}")
    with open(file_path, "wb") as f:
        f.write(await image.read())

    logger.info(f"Image uploaded successfully: {file_path}")
    return {"message": "Image uploaded successfully.", "filename": file_path}


@app.post("/caption", response_model=dict)
async def caption_upload(image: UploadFile = File(...)):
    """
    POST /caption with an image file to generate a caption immediately.
    This complements the GET /caption which operates on the last uploaded file.
    """
    logger.info("Direct caption upload endpoint called")
    if not image.filename:
        logger.warning("No file provided to caption endpoint")
        raise HTTPException(status_code=400, detail="No file selected.")

    extension = image.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        logger.warning(f"Invalid file format: {extension}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only jpg, jpeg, png are supported.",
        )

    try:
        image_bytes = await image.read()
        result = cap.generate_from_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Caption generation failed for uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # If the captioner returned a structured dict (with tokens/scores/attention), merge it into response
    if isinstance(result, dict):
        resp = {"filename": image.filename}
        resp.update(result)
        return resp
    return {"filename": image.filename, "caption": result}


@app.get("/caption", response_model=dict)
def generate_caption():
    """
    Endpoint to generate a caption for the uploaded image.
    """
    logger.info("Caption generation endpoint called")
    # Check if there's an uploaded image
    files = os.listdir(UPLOAD_FOLDER)
    if not files:
        logger.warning("No uploaded image found when attempting to generate a caption")
        raise HTTPException(
            status_code=400, detail="No image found. Please upload an image first."
        )

    image_name = files[0]  # There should be only one file in the folder
    image_path = os.path.join(UPLOAD_FOLDER, image_name)

    # Generate the caption using the bytes-based generator to get structured output
    try:
        logger.info(f"Generating caption for image: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        result = cap.generate_from_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Caption generation failed for {image_path}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Caption generation failed: {str(e)}"
        )

    logger.info(f"Caption generated successfully for {image_path}")
    # If structured dict returned, merge with filename; otherwise return simple caption
    if isinstance(result, dict):
        resp = {"image": image_name}
        resp.update(result)
        return resp
    return {"image": image_name, "caption": result}


if __name__ == "__main__":
    logger.info("Registering mDNS service...")
    register_mdns_service()

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting the API server at http://{local_ip}:{port}")
        import uvicorn

        uvicorn.run(
            app, host="0.0.0.0", port=port, log_level="info", timeout_keep_alive=5
        )
    except Exception as e:
        logger.error(f"Failed to start the API server: {e}")
    finally:
        logger.info("Shutting down the server")
        if service_info:
            logger.info("Unregistering mDNS service...")
            zeroconf.unregister_service(service_info)
            zeroconf.close()
            logger.info("mDNS service unregistered.")

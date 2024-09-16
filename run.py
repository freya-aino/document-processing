import io
import cv2
import pytesseract
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ------------------------------------------------------------------------------

# set tesceract path
pytesseract.pytesseract.tesseract_cmd  = "/usr/bin/tesseract"

# setup fastapi app
app = FastAPI()

# ------------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    
    return Image.fromarray(denoised)

@app.get("/agailable-languages")
async def get_available_languages() -> JSONResponse:
    try:
        languages = pytesseract.get_languages(config="")
        return JSONResponse(content={"languages": languages}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/extract-text")
async def extract_text(image: UploadFile = File(...), language: str = "deu") -> JSONResponse:
    try:
        raw_file = await image.read()
        image = Image.open(io.BytesIO(raw_file))
        
        image = preprocess_image(image)
        
        text = pytesseract.image_to_string(image, lang="deu", config="--psm 6")
        
        return JSONResponse(content={"text": text}, status_code=200)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

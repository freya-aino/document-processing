import io

from torch import cuda
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# ----------------------------------



app = FastAPI()

# ----------------------------------

# Explicitly set clean_up_tokenization_spaces to avoid warning
tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", clean_up_tokenization_spaces=True)

nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
    tokenizer=tokenizer,
    device="cuda" if cuda.is_available() else "cpu",
)


@app.post("/upload-file/")
async def ask_question(file: UploadFile = File(...), question: str = None):
    
    try:
        image = await file.read()
        image = Image.open(io.BytesIO(image))
        
        answer = nlp(image, question)
        
        return JSONResponse(content={"answer": answer}, status_code=200)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
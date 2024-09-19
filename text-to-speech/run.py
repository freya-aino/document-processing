from transformers import pipeline
from datasets import load_dataset
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# ---------------------------------------------------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------------------------------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/tts")
def tts(text: str):
    try:
        if text == "":
            return JSONResponse(content={"error": "Text is empty"}, status_code=400)
        
        speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
        return JSONResponse(content={
            "audio": speech["audio"].tolist(),
            "sample_rate": speech["sampling_rate"]
        }, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
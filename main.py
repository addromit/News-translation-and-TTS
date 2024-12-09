import os
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from translate_and_speakp import translate_and_tts, language_mapping

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class TranslationRequest(BaseModel):
    url: str
    lang: str

@app.get("/")
async def serve_index():
    """Serve the index.html file."""
    index_file_path = os.path.join("static", "index.html")
    if not os.path.exists(index_file_path):
        raise HTTPException(status_code=404, detail="Index file not found.")
    return FileResponse(index_file_path)

@app.post("/translate")
async def translate(request: TranslationRequest):
    """Translate a news article and generate audio."""
    try:
        url = request.url.strip()
        lang = request.lang.strip()

        # Validate inputs
        if not url:
            raise HTTPException(status_code=400, detail="URL is required.")
        if not lang:
            raise HTTPException(status_code=400, detail="Language is required.")
        if lang not in language_mapping:
            supported_languages = list(language_mapping.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported language. Supported languages are: {', '.join(supported_languages)}"
            )

        print(f"Received URL: {request.url}, Language: {request.lang}")# Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running translation and TTS on device: {device}")

        # Run translation and TTS
        translated_text, audio_file = translate_and_tts(lang, url, device=device)
        print(f"Audio file generated at: {audio_file}")

        if not os.path.exists(audio_file):
            raise HTTPException(status_code=500, detail="Audio file generation failed.")

        # Return success response
        return {
            "translated_text": translated_text,
            "audio_file": audio_file.split('/')[-1]  # Return only the file name
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in translation endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve the generated audio file."""
    try:
        # Construct full file path
        audio_path = os.path.join("static", "audio", filename)

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")

        return FileResponse(audio_path, media_type="audio/wav", filename=filename)

    except Exception as e:
        print(f"Error retrieving audio file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving audio.")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "supported_languages": list(language_mapping.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

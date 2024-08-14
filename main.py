import logging
from fastapi import FastAPI, Query, Response
from fastapi.staticfiles import StaticFiles
import logging
from fastapi.responses import FileResponse, JSONResponse  # Import JSONResponse
import io
from translator import translate_article_from_url
from TTS import TextToSpeech

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    logging.info("Serving index.html")
    return FileResponse("static/index.html")

@app.get("/translate")
async def translate(url: str, lang: str):
    logging.info(f"Received translation request for URL: {url} with target language: {lang}")
    
    target_lang = lang
    logging.info(f"Target language code resolved to: {target_lang}")
    
    try:
        en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
        logging.info("Starting translation...")
        translation = translate_article_from_url(url, en_indic_ckpt_dir, "en-indic", src_lang="eng_Latn", tgt_lang=target_lang)
        logging.info("Translation completed successfully.")
        logging.info(JSONResponse(content={"translation": translation}))
        return JSONResponse(content={"translation": translation})
    
    except Exception as e:
        logging.error(f"Error during translation: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


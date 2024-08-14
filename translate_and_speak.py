from translator import Translator
from TTS import TextToSpeech

def translate_and_speak(url, src_lang, tgt_lang, translation_ckpt_dir, tts_lang_code, direction="en-indic", quantization=None):
    translator = Translator(translation_ckpt_dir, direction, quantization)
    translator.translate_article_from_url(url, src_lang, tgt_lang)
    article_text = translator.scrape_article(url)
    if article_text:
        sentences = translator.process_article(article_text)
        translated_text = " ".join(translator.batch_translate(sentences, src_lang, tgt_lang))
        tts = TextToSpeech(tts_lang_code)
        audio_data = tts.synthesize_text(translated_text)
        tts.play_audio()
        return translated_text, audio_data
    else:
        print("Failed to retrieve article.")
        return None, None

if __name__ == "__main__":
    url = input("Enter the URL of the article: ")
    src_lang = input("Enter the source language code (e.g., eng_Latn): ")
    tgt_lang = input(f"Enter the target language code: ")
    tts_lang_code = input(f"Enter the language code for TTS: ")
    translation_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    translated_text, audio_data = translate_and_speak(url, src_lang, tgt_lang, translation_ckpt_dir, tts_lang_code)
    if translated_text and audio_data:
        print(f"Translated Text:\n{translated_text}")


import os
import sys
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from IndicTransToolkit.processor import IndicProcessor
import bitsandbytes as bnb
import torch
from scipy.io.wavfile import write
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import nltk
from newspaper import Article
import os
import subprocess
import locale
from models import SynthesizerTrn
import subprocess
import tempfile
# from IPython.display import Audio
import re
import glob
import json
import tempfile
import math
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import bitsandbytes as bnb

os.environ['NUMBA_DISABLE_CACHING'] = '1'
locale.getpreferredencoding = lambda: "UTF-8"
nltk.download('punkt')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8


# Set the directories for vits and IndicTransToolkit
vits_dir = os.path.join(os.getcwd(), 'app/vits')  # Directory for vits
indic_dir = os.path.join(os.getcwd(), 'app/IndicTransToolkit')  # Directory for IndicTransToolkit
indicnlp_dir = os.path.join(os.getcwd(), 'app/indic_nlp_library')
# Append the directories to sys.path
sys.path.append(vits_dir)
sys.path.append(indic_dir)
sys.path.append(indicnlp_dir)

def extract_article(url):
    """
    Extracts title and text from the URL and splits into sentences.
    """
    article = Article(url)
    article.download()
    article.parse()
    article_text = f"{article.title}\n\n{article.text}"
    return nltk.sent_tokenize(article_text)

def initialize_model_and_tokenizer(ckpt_dir="ai4bharat/indictrans2-en-indic-1B", quantization=None):
    """
    Initializes model and tokenizer with optional quantization using transformers' BitsAndBytesConfig.
    """
    qconfig = None
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True, bnb_8bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig
    ).to(DEVICE)

    if DEVICE == "cuda" and qconfig is None:
        model.half()  # Use half-precision for faster inference if no quantization
    model.eval()
    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    src_lang= "eng_Latn"

    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text

        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

def download(lang, tgt_dir="./"):
  lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
  cmd = ";".join([
        f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn}",
        f"tar zxvf {lang_fn}"
  ])
  print(f"Download model for language: {lang}")
  subprocess.check_output(cmd, shell=True)
  print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
  return lang_dir


def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == 'ron':
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, \
             tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd +=  f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line =  re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


def process_sentence(sentence, text_mapper, hps, lang, model, device):
    # Preprocess and map text
    txt = preprocess_text(sentence, text_mapper, hps, lang=lang)
    stn_tst = text_mapper.get_text(txt, hps)

    # Generate audio representation
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

        # Perform inference
        hyp = model.infer(
            x_tst, x_tst_lengths, noise_scale=0.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0, 0].cpu().float().numpy()

    return hyp

from pydub import AudioSegment

def aggregate_audio(translation, text_mapper, hps, lang, model, device):
    try:
        audio_segments = []

        if not translation or len(translation) == 0:
            print("Translation is empty or invalid.")
            raise ValueError("Translation input is empty.")

        for idx, sentence in enumerate(translation):
            print(f"Processing sentence {idx + 1}: {sentence}")
            hyp_audio = process_sentence(sentence, text_mapper, hps, lang, model, device)
            if hyp_audio is None or len(hyp_audio) == 0:
                print(f"Failed to generate audio for sentence {idx + 1}: {sentence}")
            else:
                audio_segments.append(hyp_audio)

        if not audio_segments:
            print("No audio segments were generated.")
            raise ValueError("No audio segments generated.")

        aggregated_audio = np.concatenate(audio_segments)
        os.makedirs('static/audio', exist_ok=True)

        # Save as WAV first
        wav_path = f"static/audio/{lang}_aggregated_audio.wav"
        write(wav_path, hps.data.sampling_rate, aggregated_audio.astype(np.float32))

        # Convert WAV to MP3
        mp3_path = f"static/audio/{lang}_aggregated_audio.mp3"
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")

        print(f"Generated and saved audio to {mp3_path}")
        return mp3_path
    except Exception as e:
        print(f"Error in aggregate_audio: {e}")
        return None

language_mapping = {
    "Assamese": {"tts_code": "asm", "indictrans_code": "asm_Beng"},
    "Bengali": {"tts_code": "ben", "indictrans_code": "ben_Beng"},
    "Gujarati": {"tts_code": "guj", "indictrans_code": "guj_Gujr"},
    "Hindi": {"tts_code": "hin", "indictrans_code": "hin_Deva"},
    "Kannada": {"tts_code": "kan", "indictrans_code": "kan_Knda"},
    "Malayalam": {"tts_code": "mal", "indictrans_code": "mal_Mlym"},
    "Marathi": {"tts_code": "mar", "indictrans_code": "mar_Deva"},
    "Nepali": {"tts_code": "nep", "indictrans_code": "npi_Deva"},
    "Odia": {"tts_code": "ori", "indictrans_code": "ory_Orya"},
    "Punjabi, Eastern": {"tts_code": "pan", "indictrans_code": "pan_Guru"},
    "Tamil": {"tts_code": "tam", "indictrans_code": "tam_Taml"},
    "Telugu": {"tts_code": "tel", "indictrans_code": "tel_Telu"},
    "Urdu": {"tts_code": "urd", "indictrans_code": "urd_Arab"},
    "English": {"tts_code": "eng", "indictrans_code": "eng_Latn"}
}

def get_language_codes(language_name):
    language_info = language_mapping.get(language_name)
    if language_info:
        return language_info
    else:
        return {"error": "Language not found in mapping."}

def translate_and_tts(language_name, url, device="cpu"):
    # Retrieve language codes
    print(f"Language received: {language_name}")

    language_info = get_language_codes(language_name)
    if "error" in language_info:
        raise ValueError(language_info["error"])

    LANG = language_info["tts_code"]
    tgt_lang = language_info["indictrans_code"]

    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
        en_indic_ckpt_dir, quantization="8-bit" if device == "cuda" else None
    )
    ip = IndicProcessor(inference=True)

    # Convert string to torch device for consistency
    device = torch.device(device)  # Set torch device here
    print(f"Running inference with {device}")

    # Load model files
    ckpt_dir = download(LANG)
    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    )
    net_g.to(device)  # Ensure model is moved to the correct device
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"Loading {g_pth}")
    _ = utils.load_checkpoint(g_pth, net_g, None)

    # Extract and translate article text
    article = extract_article(url)
    article = " ".join(article)
    print("Original Article:", article)

    translated_text = batch_translate(
        [article], src_lang="eng_Latn", tgt_lang=tgt_lang, model=en_indic_model, tokenizer=en_indic_tokenizer, ip=ip
    )[0]
    print("Translated Text:", translated_text)

    # Preprocess text for TTS
    preprocessed_text = preprocess_text(translated_text, text_mapper, hps, uroman_dir=None, lang=LANG)

    # Generate and aggregate audio
    audio_file_path = aggregate_audio([preprocessed_text], text_mapper, hps, LANG, net_g, device)

    # Ensure audio file is correctly generated
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise ValueError("Failed to generate audio file.")

    print(f"Audio file generated successfully: {audio_file_path}")

    return translated_text, audio_file_path


# def translate_and_tts(language_name, url, device="cpu"):
#     # Retrieve language codes
#     print(f"Language received: {language_name}")

#     language_info = get_language_codes(language_name)
#     if "error" in language_info:
#         raise ValueError(language_info["error"])

#     LANG = language_info["tts_code"]
#     tgt_lang = language_info["indictrans_code"]

#     en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
#     en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(
#         en_indic_ckpt_dir, quantization="8-bit" if device == "cuda" else None
#     )
#     ip = IndicProcessor(inference=True)

#     # Convert string to torch device for consistency
#     device = torch.device(device)  # Set torch device here
#     print(f"Running inference with {device}")

#     # Load model files
#     ckpt_dir = download(LANG)
#     vocab_file = f"{ckpt_dir}/vocab.txt"
#     config_file = f"{ckpt_dir}/config.json"
#     assert os.path.isfile(config_file), f"{config_file} doesn't exist"
#     hps = utils.get_hparams_from_file(config_file)
#     text_mapper = TextMapper(vocab_file)
#     net_g = SynthesizerTrn(
#         len(text_mapper.symbols),
#         hps.data.filter_length // 2 + 1,
#         hps.train.segment_size // hps.data.hop_length,
#         **hps.model
#     )
#     net_g.to(device)  # Ensure model is moved to the correct device
#     _ = net_g.eval()

#     g_pth = f"{ckpt_dir}/G_100000.pth"
#     print(f"Loading {g_pth}")
#     _ = utils.load_checkpoint(g_pth, net_g, None)

#     # Extract and translate article text
#     article = extract_article(url)
#     article = " ".join(article)
#     print("Original Article:", article)

#     translated_text = batch_translate(
#         [article], src_lang="eng_Latn", tgt_lang=tgt_lang, model=en_indic_model, tokenizer=en_indic_tokenizer, ip=ip
#     )[0]
#     print("Translated Text:", translated_text)

#     # Preprocess text for TTS
#     text_mapper = TextMapper(vocab_file=vocab_file)
#     preprocessed_text = preprocess_text(translated_text, text_mapper, hps, uroman_dir=None, lang=LANG)

#     # Generate and aggregate audio
#     audio = process_sentence(preprocessed_text, text_mapper, hps, LANG, net_g, device)  # Pass device here
#     audio_file_path = aggregate_audio([preprocessed_text], text_mapper, hps, LANG, net_g, device)  # Pass device here

#     # Save audio file to static/audio
#     audio_file_path = f"static/audio/{LANG}_audio.wav"
#     os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)  # Ensure the directory exists

#     with open(audio_file_path, 'wb') as f:
#         f.write(audio)  # Assuming audio is in binary format

#     return translated_text, audio_file_path





def aggregate_audio(translation, text_mapper, hps, lang, model, device):
    try:
        audio_segments = []

        if not translation or len(translation) == 0:
            print("Translation is empty or invalid.")
            raise ValueError("Translation input is empty.")

        for idx, sentence in enumerate(translation):
            print(f"Processing sentence {idx + 1}: {sentence}")
            hyp_audio = process_sentence(sentence, text_mapper, hps, lang, model, device)
            if hyp_audio is None or len(hyp_audio) == 0:
                print(f"Failed to generate audio for sentence {idx + 1}: {sentence}")
            else:
                audio_segments.append(hyp_audio)

        if not audio_segments:
            print("No audio segments were generated.")
            raise ValueError("No audio segments generated.")

        aggregated_audio = np.concatenate(audio_segments)
        os.makedirs('static/audio', exist_ok=True)
        output_path = f"static/audio/{lang}_aggregated_audio.wav"
        write(output_path, hps.data.sampling_rate, aggregated_audio.astype(np.float32))

        print(f"Generated and saved audio to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in aggregate_audio: {e}")
        return None


# Ensure supporting directories are correctly set
def download(lang, tgt_dir="./"):
    lang_fn = os.path.join(tgt_dir, f"{lang}.tar.gz")
    lang_dir = os.path.join(tgt_dir, lang)
    os.makedirs(tgt_dir, exist_ok=True)
    if not os.path.exists(lang_fn):
        cmd = f"wget https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz -O {lang_fn} && tar -xvzf {lang_fn} -C {tgt_dir}"
        subprocess.run(cmd, shell=True, check=True)
    return lang_dir
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import torch
import newspaper
import re

class Translator:
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, ckpt_dir, direction, quantization=None):
        self.ckpt_dir = ckpt_dir
        self.direction = direction
        self.quantization = quantization
        self.tokenizer, self.model = self.initialize_model_and_tokenizer()

    def initialize_model_and_tokenizer(self):
        tokenizer = IndicTransTokenizer(direction=self.direction)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.ckpt_dir,
            trust_remote_code=True,
            quantization_config=self.quantization,
        )

        if self.quantization is None:
            model = model.to(self.DEVICE)
            if self.DEVICE == "cuda":
                model.half()

        model.eval()
        return tokenizer, model

    def batch_translate(self, input_sentences, src_lang, tgt_lang):
        translations = []
        ip = IndicProcessor(inference=True)
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i : i + self.BATCH_SIZE]
            batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

            inputs = self.tokenizer(
                batch,
                src=True,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.DEVICE)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            generated_tokens = self.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
            translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations

    @staticmethod
    def process_article(article):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', article.strip())
        return sentences

    @staticmethod
    def scrape_article(url):
        article = newspaper.Article(url)
        try:
            article.download()
            article.parse()
            return article.text.strip()
        except Exception as e:
            print(f"Error fetching article: {str(e)}")
            return None

    def translate_article_from_url(self, url, src_lang="eng_Latn", tgt_lang="{tgt_lang}"):
        article_text = self.scrape_article(url)
        if article_text:
            sentences = self.process_article(article_text)
            translations = self.batch_translate(sentences, src_lang, tgt_lang)

            print(f"\n{src_lang} - {tgt_lang}")
            translation = " ".join(translations)
            print(f"{tgt_lang}: {translation}")

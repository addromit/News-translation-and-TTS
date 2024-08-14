import pickle
from ttsmms import TTS
from pydub import AudioSegment
from pydub.playback import play

class TextToSpeech:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.tts = TTS(f"data/{lang_code}")
        self.wav_text = None

    def synthesize_text(self, text):
        self.wav_text = self.tts.synthesis(text)
        return self.wav_text["x"]

    def play_audio(self):
        if self.wav_text:
            audio_data = AudioSegment(
                data=self.wav_text["x"],
                sample_width=self.wav_text.get("sample_width", 2),
                frame_rate=self.wav_text.get("sampling_rate", 16000),
                channels=self.wav_text.get("channels", 1)
            )
            play(audio_data)
        else:
            print("No audio to play. Please synthesize text first.")

    def save_model(self, filepath):
        state_dict = {
            'lang_code': self.lang_code,
            'wav_text': self.wav_text if self.wav_text else None
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)

    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)

        tts_instance = cls(state_dict['lang_code'])
        if state_dict['wav_text']:
            tts_instance.wav_text = state_dict['wav_text']

        return tts_instance

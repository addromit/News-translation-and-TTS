Here’s a detailed `README.md` file for your **News Translation and TTS** project:

---

# News Translation and TTS

This project is a **News Article Translation and Text-to-Speech (TTS)** application that allows users to translate English news articles into multiple Indian languages and listen to the translated text. Built with cutting-edge technologies like **IndicTrans**, **TTSMMS**, and **Docker**, this application is ideal for breaking language barriers in news consumption.

---

## 🚀 Features

1. **Translation**:
   - Translate English news articles into Indian languages such as Hindi, Marathi, Telugu, Gujarati, and more using the **IndicTrans2 model**.
   
2. **Text-to-Speech**:
   - Generate audio for translated articles using the **TTSMMS model**.
   
3. **Responsive Interface**:
   - Modern, user-friendly interface with dynamic updates for results and playback.
   
4. **Dockerized Deployment**:
   - Fully containerized setup for easy deployment across different environments.

---

## 🛠️ Technologies Used

- **Backend**: Python, FastAPI
- **Frontend**: HTML, CSS
- **Machine Translation**: IndicTrans2 (ai4bharat)
- **Text-to-Speech (TTS)**: TTSMMS
- **Audio Processing**: FFmpeg, Pydub
- **Deployment**: Docker
- **Libraries**: Transformers, Torch, NLTK, Pydub, Sentencepiece

---

## 📂 Project Structure

```plaintext
News-translation-and-TTS/
├── Dockerfile               # Docker configuration for deployment
├── main.py                  # FastAPI app entry point
├── requirements.txt         # Python dependencies
├── static/
│   ├── audio/               # TTS audio files
│   ├── index.html           # Frontend HTML
│   ├── styles.css           # Frontend styles
├── translate_and_speakp.py  # Translation and TTS core script
├── hin_aggregated_audio.wav # Sample TTS audio
```

---

## 🧑‍💻 How to Run the Project

### Prerequisites
- Install **Docker** on your system.
- Clone this repository:
  ```bash
  git clone https://github.com/romit-23/News-translation-and-TTS.git
  cd News-translation-and-TTS
  ```

### Run with Docker
1. Build the Docker image:
   ```bash
   docker build -t news-translation-tts .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 news-translation-tts
   ```
3. Open the application in your browser at:  
   **http://localhost:8000**

---

## 📝 Usage Instructions

1. Enter the URL of a news article in the input box.
2. Select a target language from the dropdown menu.
3. Click **Translate** to view the translated text.
4. Click the **Play Audio** button to hear the TTS output of the translated article.

---

## 🎯 Supported Languages

- Hindi (`hi`)
- Marathi (`mr`)
- Telugu (`te`)
- Gujarati (`gu`)
- Kannada (`kn`)
- Malayalam (`ml`)
- Tamil (`ta`)
- Punjabi (`pa`)
- Bengali (`bn`)

---

## 📊 API Endpoints

| **Endpoint**         | **Method** | **Description**                 |
|-----------------------|------------|---------------------------------|
| `/translate`          | POST       | Translate text into a language |
| `/tts`                | POST       | Generate TTS for translated text |

---

## 💡 Key Highlights

1. **IndicTrans2 for Machine Translation**: Achieves high-quality translations for Indian languages.
2. **TTSMMS for Speech Synthesis**: Offers natural-sounding voice output for multiple Indian languages.
3. **Dynamic Frontend**: Displays results instantly and prevents layout shifts.

---

## 🐛 Known Issues

- Long articles may take more time for processing translation and TTS.
- Audio playback might not work on some browsers due to codec compatibility. Use Chrome for best results.

---

## 🌐 Future Improvements

- Add support for more Indian languages.
- Improve TTS models for faster response time.
- Enhance the user interface with progress indicators.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🧑‍💻 Author

Developed by **Romit Addagatla**  
[GitHub Profile](https://github.com/romit-23)  

For any questions or suggestions, feel free to reach out! 

--- 

This `README.md` provides a detailed overview and serves as documentation for your repository. Let me know if you'd like additional changes!

AUDICTEE is a full-stack web application for French listening and grammar practice.  
在我学习法语的痛苦路上构思的网页，但目前很多功能和判断方法很简单粗暴



have a try：**[Audictée](https://audictee-air.vercel.app)**


由于免费计划限制，后端部署失败，因此我在localhost展示一下使用效果




## Demo

<p align="center">
  <img src="assets/demo1.jpg" width="32%">
  <img src="assets/demo2.jpg" width="32%">
  <img src="assets/demo3.jpg" width="32%">
</p>





# For youuuuu

# 🎧 YouTube Transcript → AI French Learning

Full-stack application that turns YouTube videos into interactive French learning exercises using NLP and deep learning.

Supports:
- 📡 YouTube official subtitles (API)
- 🧠 Local Whisper ASR transcription

---

## ✨ Features

### Transcript Processing
- Fetch or transcribe video audio
- Clean noise
- Merge into sentence-level structure

### Linguistic Analysis
Powered by spaCy:
- Tokenization
- Lemma extraction
- POS tagging
- Morphological features
- Character offsets

### Interactive Modes
- **Grammar Mode**: Click any word to inspect lemma, POS, morphology  
- **Fill-in-the-Blank**: Hide NOUN/VERB tokens with validation  
- **Difficulty Scoring**: Heuristic sentence complexity  
- **Semantic Similarity**: Sentence embeddings + cosine similarity  

---

## 🏗 Stack

**Backend**
- FastAPI
- spaCy (`fr_core_news_sm`)
- SentenceTransformers (MiniLM)
- NumPy
- yt-dlp
- faster-whisper

**Frontend**
- Vue 3
- Vite
- Axios

---

## Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
If using Whisper:

brew install ffmpeg
```bash
brew install ffmpeg
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
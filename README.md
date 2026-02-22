AUDICTEE is a full-stack web application for French listening and grammar practice.  
在我学习法语的痛苦路上构思的网页，但目前很多功能和判断方法很简单粗暴
以下是非我本人撰写的介绍：
not written by me：

## Features (Current Stable Version)

### YouTube Transcript Loader
- Paste a YouTube URL
- Automatically fetch transcript
- Merge segments into readable sentences

### Linguistic Analysis (spaCy)
Each sentence is processed with:
- Tokenization
- Lemma extraction
- POS tagging
- Morphological features
- Character offsets
- Punctuation detection

###  Grammar Mode
- Click a word to display:
  - Text
  - Lemma
  - POS
  - Morphological attributes

### Fill-in-the-Blank Mode
- Randomly hides NOUN and VERB tokens
- User input validation
- Accuracy tracking
- Wrong word memory system

### Sentence Difficulty Scoring
Heuristic difficulty formula based on:
- Sentence length
- POS composition

### Semantic Similarity (Deep Learning)
- Sentence embeddings via `sentence-transformers`
- Cosine similarity via vector dot product
- Display most similar sentence interactively

---

## Tech Stack

### Backend
- Python
- FastAPI
- spaCy (`fr_core_news_sm`)
- SentenceTransformers (MiniLM multilingual)
- NumPy

### Frontend
- Vue 3 (Composition API)
- Vite
- Axios


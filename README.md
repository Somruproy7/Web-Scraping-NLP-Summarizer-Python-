# Web Scraping & NLP Text Analyzer

## 📌 Overview
This project is a **Python-based web scraping and natural language processing (NLP) application** that extracts textual content from websites, analyzes sentiment and emotions, and generates concise summaries using **Hugging Face transformer models**.

It demonstrates an **end-to-end backend pipeline**, starting from web data extraction (scraping) to unstructured text processing and AI-based summarization.

---

## 🚀 Features
- 🌐 **Web Scraping**
  - Extracts visible text from static HTML websites
  - Removes scripts, styles, and irrelevant content
  - Uses custom User-Agent headers

- 🧠 **Natural Language Processing**
  - Tokenization and preprocessing with NLTK
  - Sentiment analysis using VADER
  - Emotion detection using keyword-based scoring

- ✨ **Text Summarization**
  - Abstractive summarization using **Hugging Face BART model**
  - Fallback extractive summarization for robustness
  - Handles long-form content safely

- 🧩 **Backend-Oriented Design**
  - Modular, class-based Python architecture
  - Easily extendable to FastAPI/Django services
  - Suitable for scheduled or batch scraping workflows

---

## 🛠 Tech Stack
- **Language:** Python  
- **Web Scraping:** Requests, BeautifulSoup  
- **NLP:** NLTK, Hugging Face Transformers  
- **Model:** facebook/bart-large-cnn  
- **Sentiment Analysis:** VADER  

---

## 📂 Project Structure
```
├── web_scraper_analyzer.py
├── README.md
├── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Somruproy7/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install nltk transformers torch requests beautifulsoup4
```

---

## ▶️ How to Run
```bash
python web_scraper_analyzer.py
```

Enter a website URL when prompted:
```
https://example.com
```

---

## 📊 Output
- Total word count of scraped content  
- Primary detected emotion  
- Emotion distribution percentages  
- AI-generated summary of webpage content  

---

## 🧠 Use Cases
- Web content summarization
- Market research & content analysis
- Automated document intelligence
- Data preprocessing for ML pipelines
- Backend scraping & analytics systems

---

## ⚠️ Limitations
- Supports **static HTML websites only**
- JavaScript-rendered sites require Playwright or Selenium
- Designed for ethical and compliant data access

---

## 🔮 Future Enhancements
- JavaScript-rendered site scraping (Playwright)
- Proxy rotation & rate limiting
- FastAPI/Django REST API integration
- Database storage (PostgreSQL / MongoDB)
- Scheduled scraping (cron / Airflow)

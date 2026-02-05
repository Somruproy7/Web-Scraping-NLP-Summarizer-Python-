# -*- coding: utf-8 -*-

import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

class WebScrapingTextAnalyzer:
    """
    A web scraping + NLP analysis tool.
    Scrapes text from a webpage, analyzes sentiment/emotion,
    and generates a summary using Hugging Face transformers.
    """

    EMOTION_KEYWORDS = {
        'joy': ['happy', 'joy', 'delight', 'smile', 'pleasure', 'excited'],
        'sadness': ['sad', 'grief', 'cry', 'depressed', 'lonely'],
        'anger': ['angry', 'rage', 'hate', 'furious'],
        'fear': ['fear', 'panic', 'anxious', 'terrified'],
        'love': ['love', 'affection', 'romantic'],
        'hope': ['hope', 'optimistic', 'dream']
    }

    def __init__(self, url: str):
        self.url = url
        self.text = self.scrape_website()
        self.words = self._get_words()
        self.sia = SentimentIntensityAnalyzer()
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # ---------------- SCRAPING ---------------- #
    def scrape_website(self) -> str:
        """
        Scrape visible text content from a static website.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"
            }
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts and styles
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator=" ")
            text = re.sub(r"\s+", " ", text).strip()
            return text

        except Exception as e:
            raise RuntimeError(f"Failed to scrape website: {str(e)}")

    # ---------------- NLP ---------------- #
    def _get_words(self) -> List[str]:
        try:
            return word_tokenize(self.text.lower())
        except Exception:
            return re.findall(r'\b\w+\b', self.text.lower())

    def get_word_count(self) -> int:
        return len(self.words)

    def analyze_emotion(self) -> Tuple[str, Dict[str, float]]:
        emotion_scores = {emotion: 0 for emotion in self.EMOTION_KEYWORDS}

        for word in self.words:
            for emotion, keywords in self.EMOTION_KEYWORDS.items():
                if word in keywords:
                    emotion_scores[emotion] += 1

        sentiment = self.sia.polarity_scores(self.text)
        if sentiment["compound"] > 0.3:
            emotion_scores["joy"] += 2
            emotion_scores["hope"] += 1
        elif sentiment["compound"] < -0.3:
            emotion_scores["sadness"] += 1
            emotion_scores["anger"] += 1

        total = sum(emotion_scores.values()) or 1
        percentages = {k: (v / total) * 100 for k, v in emotion_scores.items()}
        primary = max(emotion_scores, key=emotion_scores.get)

        return primary, percentages

    def summarize(self) -> str:
        """
        Generate summary using Hugging Face model.
        """
        try:
            if len(self.text.split()) < 100:
                return "Text too short for meaningful summarization."

            summary = self.summarizer(
                self.text[:4000],  # prevent token overflow
                max_length=150,
                min_length=40,
                do_sample=False
            )
            return summary[0]["summary_text"]

        except Exception:
            # Fallback extractive summary
            sentences = sent_tokenize(self.text)
            return " ".join(sentences[:3])

# ---------------- MAIN ---------------- #
def main():
    print("Enter a website URL to scrape and analyze:")
    url = input().strip()

    try:
        analyzer = WebScrapingTextAnalyzer(url)

        print("\nAnalysis Results")
        print("-" * 50)
        print(f"Word Count: {analyzer.get_word_count()}")

        emotion, breakdown = analyzer.analyze_emotion()
        print(f"\nPrimary Emotion: {emotion}")
        print("Emotion Breakdown:")
        for k, v in breakdown.items():
            if v > 0:
                print(f"- {k.title()}: {v:.1f}%")

        print("\nSummary:")
        print(analyzer.summarize())

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

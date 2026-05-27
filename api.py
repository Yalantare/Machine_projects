import sys
import os
import re
import string
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any

# Custom tokenizer needed for loading the model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian")
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['фильм', 'сюжет', 'кино', 'картина', 'режиссер', 'роль', 'актер'])

def custom_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        stemmer.stem(token) for token in tokens 
        if token not in russian_stopwords and len(token) > 2
    ]
    return cleaned_tokens

# CRITICAL PICKLE FIX: Inject custom_tokenizer into sys.modules['__main__']
# This allows joblib.load to find it regardless of how the script was launched (via uvicorn, runpy, etc.)
sys.modules['__main__'].custom_tokenizer = custom_tokenizer

app = FastAPI(
    title="Kinopoisk Movie Genre API",
    description="API для определения жанров фильмов по описанию сюжета",
    version="1.0.0"
)

# Load the trained model pipeline
model_path = os.path.join(os.path.dirname(__file__), "movie_classifier.joblib")
if not os.path.exists(model_path):
    # Fallback to absolute temp path
    model_path = r"C:\Users\User\AppData\Local\Temp\movie_classifier.joblib"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    model = None
    print("WARNING: Model file not found!")

# Find target workspace directory dynamically
workspace_dir = None
for name in os.listdir(r"C:\Users\User\anacond"):
    if name.startswith("3"):
        workspace_dir = os.path.join(r"C:\Users\User\anacond", name)
        break

df_cache = None
if workspace_dir:
    csv_path = os.path.join(workspace_dir, "kinopoisk_top250.csv")
    if os.path.exists(csv_path):
        df_cache = pd.read_csv(csv_path)

class MovieRequest(BaseModel):
    title: str = ""
    description: str

class PredictResponse(BaseModel):
    title: str
    predicted_genre: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict_genre(request: MovieRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Модель машинного обучения не загружена!")
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Описание фильма не должно быть пустым!")
    
    try:
        # Perform prediction
        pred_genre = model.predict([request.description])[0]
        probs = model.predict_proba([request.description])[0]
        confidence = float(np.max(probs))
        
        return PredictResponse(
            title=request.title or "Неизвестный фильм",
            predicted_genre=pred_genre.capitalize(),
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")

@app.get("/stats")
def get_stats() -> Dict[str, Any]:
    if df_cache is None:
        raise HTTPException(status_code=500, detail="Файл датасета kinopoisk_top250.csv не найден!")
    
    try:
        df = df_cache.copy()
        df['description'] = df['description'].fillna('').astype(str)
        df['genre'] = df['genre'].fillna('').astype(str)
        
        # Calculations
        total_movies = len(df)
        char_lengths = df['description'].apply(len)
        word_lengths = df['description'].apply(lambda x: len(x.split()))
        
        primary_genres = df['genre'].apply(lambda x: x.split(',')[0].strip().lower())
        genre_distribution = primary_genres.value_counts().to_dict()
        genre_share = (primary_genres.value_counts(normalize=True) * 100).round(2).to_dict()
        
        rating_mean = float(df['rating'].mean())
        rating_min = float(df['rating'].min())
        rating_max = float(df['rating'].max())
        
        year_distribution = df['year'].value_counts().sort_index().to_dict()
        
        return {
            "total_movies": total_movies,
            "avg_description_chars": round(float(char_lengths.mean()), 1),
            "avg_description_words": round(float(word_lengths.mean()), 1),
            "genre_counts": genre_distribution,
            "genre_shares_percent": genre_share,
            "ratings": {
                "mean": round(rating_mean, 2),
                "min": rating_min,
                "max": rating_max
            },
            "years": year_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка расчета статистики: {str(e)}")

@app.get("/help")
def get_help() -> Dict[str, Any]:
    return {
        "service_name": "API определения категории фильмов Кинопоиска",
        "description": "Программный интерфейс (API) для классификации сюжета фильма по жанрам.",
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Предсказать основной жанр фильма по его текстовому описанию.",
                "request_body": {
                    "title": "Название фильма (строка, необязательно)",
                    "description": "Сюжет фильма или его краткое описание на русском языке (строка, обязательно)"
                },
                "response": {
                    "title": "Название фильма",
                    "predicted_genre": "Предсказанный жанр",
                    "confidence": "Уверенность модели (от 0.0 до 1.0)"
                }
            },
            {
                "path": "/stats",
                "method": "GET",
                "description": "Получить агрегированные статистические данные по датасету (250 лучших фильмов).",
                "response": "Словарь со статистическими метриками (общие показатели, доли жанров, годы, рейтинги)"
            },
            {
                "path": "/help",
                "method": "GET",
                "description": "Информационная справка по имеющимся эндпоинтам и параметрам.",
                "response": "Справочное руководство"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

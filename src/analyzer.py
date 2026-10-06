import re
from typing import Any

from .text_preprocessing import preprocess_and_analyze
from .ml_pipeline import FakeNewsDetector

# Initialize ML models
ml_detector = FakeNewsDetector()

try:
    ml_detector.load_models("models")
except:
    print("Warning: Could not load pre-trained models. Automatically training models now...")
    try:
        # Автоматично извикваме функцията за обучение от твоето CSV
        ml_detector.train("news_dataset.csv")
    except Exception as e:
        print(f"Error during automatic training: {e}")


def analyze_text(text: str) -> dict[str, Any]:
    """
    Анализира подадения текст и връща:
    - основна статистика;
    - почистен текст;
    - обработени токени;
    - рискова оценка;
    - открити рискови индикатори.
    - ML прогноза за фалшиви новини
    """
    if not isinstance(text, str):
        raise TypeError("Текстът трябва да бъде от тип str.")

    analysis = preprocess_and_analyze(text)

    words = re.findall(
        r"[А-Яа-яЍѝA-Za-z]+",
        text,
    )

    tokens = analysis["tokens"]
    risk = analysis["risk"]

    # Get ML predictions
    ml_predictions = {
        'naive_bayes': ml_detector.predict(text, 'naive_bayes'),
        'svm': ml_detector.predict(text, 'svm'),
        'neural_net': ml_detector.predict(text, 'neural_net')
    }

    return {
        "word_count": len(words),
        "filtered_word_count": len(tokens),
        "character_count": len(text),
        "preview": " ".join(words[:20]),
        "cleaned_text": analysis["cleaned_text"],
        "tokens": tokens,
        "risk_score": risk["score"],
        "risk_level": risk["level"],
        "risk_messages": risk["messages"],
        "risk_details": risk,
        "ml_predictions": ml_predictions
    }

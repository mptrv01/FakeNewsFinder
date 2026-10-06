import re
from typing import Any

import spacy


# Българска езикова конфигурация на spaCy.
# Съдържа tokenizer и списък със стоп думи,
# но няма предварително обучен модел за лематизация.
nlp = spacy.blank("bg")


SENSATIONAL_PHRASES = {
    "шок",
    "шокиращо",
    "сензация",
    "скандал",
    "невероятно",
    "истината която крият",
    "няма да повярвате",
    "разтърсващо",
    "бомба",
    "ужас",
    "тайна",
    "забранено",
}


URGENCY_PHRASES = {
    "веднага",
    "спешно",
    "докато не е изтрито",
    "преди да го свалят",
    "преди да го изтрият",
    "само днес",
    "последен шанс",
}


SHARING_PHRASES = {
    "сподели",
    "споделете",
    "разпространи",
    "разпространете",
    "изпрати на всички",
    "изпратете на всички",
    "нека всички разберат",
}


ANONYMOUS_SOURCE_PHRASES = {
    "според запознати",
    "анонимен източник",
    "източници твърдят",
    "експерти твърдят",
    "учени твърдят",
    "говори се",
}


SOURCE_MARKERS = {
    "източник",
    "според",
    "съобщи",
    "съобщава",
    "цитира",
    "данни на",
    "проучване на",
    "доклад на",
    "бта",
    "бнр",
    "бнт",
}


def normalize_text(text: str) -> str:
    """
    Почиства текста за последваща обработка.

    Премахва:
    - интернет адреси;
    - пунктуация;
    - специални символи;
    - повтарящи се интервали.

    Преобразува всички букви в малки.
    """
    if not isinstance(text, str):
        raise TypeError("Текстът трябва да бъде от тип str.")

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = text.lower()

    # Запазваме българските букви, латиницата,
    # цифрите и интервалите.
    text = re.sub(
        r"[^а-яѝa-z0-9\s]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> list[str]:
    """
    Разделя текста на токени и премахва:

    - пунктуацията;
    - стоп думите;
    - еднобуквените токени;
    - числа и други несловесни елементи.
    """
    cleaned_text = normalize_text(text)
    document = nlp(cleaned_text)

    tokens = [
        token.text
        for token in document
        if token.is_alpha
        and not token.is_stop
        and len(token.text) > 1
    ]

    return tokens


def find_phrase_matches(
    text: str,
    phrases: set[str],
) -> list[str]:
    """
    Намира цели рискови думи и фрази в текста.

    Проверяването за граници на думите предотвратява
    намирането на „шок“ вътре в думата „шокиращо“.
    """
    matches = []

    for phrase in phrases:
        pattern = (
            r"(?<!\w)"
            + re.escape(phrase).replace(r"\ ", r"\s+")
            + r"(?!\w)"
        )

        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(phrase)

    return sorted(matches)


def detect_risk_indicators(text: str) -> dict[str, Any]:
    """
    Открива езикови рискови индикатори.

    Резултатът е евристична оценка, а не вероятност,
    изчислена от модел за машинно обучение.
    """
    if not isinstance(text, str):
        raise TypeError("Текстът трябва да бъде от тип str.")

    if not text.strip():
        return {
            "score": 0,
            "level": "нисък",
            "messages": [],
            "uppercase_words": [],
            "uppercase_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "sensational_matches": [],
            "urgency_matches": [],
            "sharing_matches": [],
            "anonymous_source_matches": [],
            "missing_source": False,
            "has_url": False,
        }

    lowered_text = normalize_text(text)

    words = re.findall(
        r"[А-Яа-яЍѝA-Za-z]+",
        text,
    )

    uppercase_words = re.findall(
        r"\b[А-ЯЍA-Z]{3,}\b",
        text,
    )

    uppercase_ratio = (
        len(uppercase_words) / len(words)
        if words
        else 0.0
    )

    sensational_matches = find_phrase_matches(
        lowered_text,
        SENSATIONAL_PHRASES,
    )

    urgency_matches = find_phrase_matches(
        lowered_text,
        URGENCY_PHRASES,
    )

    sharing_matches = find_phrase_matches(
        lowered_text,
        SHARING_PHRASES,
    )

    anonymous_source_matches = find_phrase_matches(
        lowered_text,
        ANONYMOUS_SOURCE_PHRASES,
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    has_url = bool(
        re.search(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE,
        )
    )

    has_source_marker = any(
        marker in lowered_text
        for marker in SOURCE_MARKERS
    )

    # Проверяваме липсата на източник само при текст,
    # който съдържа поне пет думи.
    missing_source = (
        len(words) >= 5
        and not has_url
        and not has_source_marker
    )

    score = 0
    messages = []

    if len(uppercase_words) >= 3 or uppercase_ratio >= 0.30:
        score += 15
        messages.append(
            "Открита е прекомерна употреба на главни букви."
        )

    if exclamation_count >= 3:
        score += 10
        messages.append(
            "Открити са множество удивителни знаци."
        )

    if question_count >= 3:
        score += 5
        messages.append(
            "Открити са множество въпросителни знаци."
        )

    if sensational_matches:
        score += min(
            len(sensational_matches) * 6,
            24,
        )
        messages.append(
            "Открити са сензационни думи или изрази."
        )

    if urgency_matches:
        score += min(
            len(urgency_matches) * 7,
            21,
        )
        messages.append(
            "Текстът създава усещане за спешност."
        )

    if sharing_matches:
        score += min(
            len(sharing_matches) * 8,
            16,
        )
        messages.append(
            "Текстът призовава за незабавно споделяне."
        )

    if anonymous_source_matches:
        score += min(
            len(anonymous_source_matches) * 6,
            12,
        )
        messages.append(
            "Използван е неясен или анонимен източник."
        )

    if missing_source:
        score += 10
        messages.append(
            "Не е открит ясно посочен източник."
        )

    score = min(score, 100)

    if score >= 55:
        risk_level = "висок"
    elif score >= 25:
        risk_level = "среден"
    else:
        risk_level = "нисък"

    return {
        "score": score,
        "level": risk_level,
        "messages": messages,
        "uppercase_words": uppercase_words,
        "uppercase_ratio": round(uppercase_ratio, 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sensational_matches": sensational_matches,
        "urgency_matches": urgency_matches,
        "sharing_matches": sharing_matches,
        "anonymous_source_matches": (
            anonymous_source_matches
        ),
        "missing_source": missing_source,
        "has_url": has_url,
    }


def preprocess_and_analyze(text: str) -> dict[str, Any]:
    """
    Извършва пълната предварителна обработка
    и анализа на рисковите индикатори.
    """
    cleaned_text = normalize_text(text)
    tokens = tokenize_text(text)
    risk = detect_risk_indicators(text)

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "risk": risk,
    }


if __name__ == "__main__":
    example_text = (
        "ШОКИРАЩО!!! СПЕШНО СПОДЕЛЕТЕ тази новина, "
        "преди да я изтрият!"
    )

    result = preprocess_and_analyze(example_text)

    print("Почистен текст:")
    print(result["cleaned_text"])

    print("\n Токени:")
    print(result["tokens"])

    print("\n Рискова оценка:")
    print(result["risk"]["score"])

    print("\n Ниво на риск:")
    print(result["risk"]["level"])

    print("\n Открити индикатори:")
    for message in result["risk"]["messages"]:
        print("-", message)
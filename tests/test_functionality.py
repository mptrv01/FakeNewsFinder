import unittest

from app import app
from src.analyzer import analyze_text
from src.text_preprocessing import (
    detect_risk_indicators,
    normalize_text,
    preprocess_and_analyze,
    tokenize_text,
)


class TextPreprocessingTests(unittest.TestCase):
    """Тестове за предварителната обработка на текста."""

    def test_normalize_text(self):
        text = "Това е НОВИНА!!! https://example.com"

        result = normalize_text(text)

        self.assertEqual(result, "това е новина")
        self.assertNotIn("https://", result)
        self.assertNotIn("!", result)

    def test_tokenize_text(self):
        text = "Това е важна българска новина."

        tokens = tokenize_text(text)

        self.assertIsInstance(tokens, list)
        self.assertIn("важна", tokens)
        self.assertIn("българска", tokens)
        self.assertNotIn("това", tokens)
        self.assertNotIn("е", tokens)

    def test_suspicious_text_risk(self):
        text = (
            "ШОКИРАЩО!!! СПЕШНО СПОДЕЛЕТЕ тази новина, "
            "преди да я изтрият!"
        )

        risk = detect_risk_indicators(text)

        self.assertEqual(risk["score"], 56)
        self.assertEqual(risk["level"], "висок")
        self.assertGreater(len(risk["messages"]), 0)

        self.assertIn(
            "шокиращо",
            risk["sensational_matches"],
        )

        self.assertIn(
            "спешно",
            risk["urgency_matches"],
        )

        self.assertIn(
            "споделете",
            risk["sharing_matches"],
        )

    def test_empty_text_risk(self):
        risk = detect_risk_indicators("")

        self.assertEqual(risk["score"], 0)
        self.assertEqual(risk["level"], "нисък")
        self.assertEqual(risk["messages"], [])

    def test_url_detection(self):
        text = (
            "Според БТА информацията е публикувана на "
            "https://example.com/news"
        )

        risk = detect_risk_indicators(text)

        self.assertTrue(risk["has_url"])
        self.assertFalse(risk["missing_source"])

    def test_complete_preprocessing_result(self):
        result = preprocess_and_analyze(
            "Това е примерен текст на новина."
        )

        self.assertIn("original_text", result)
        self.assertIn("cleaned_text", result)
        self.assertIn("tokens", result)
        self.assertIn("risk", result)


class AnalyzerTests(unittest.TestCase):
    """Тестове за общия анализатор."""

    def test_analyzer_result_structure(self):
        result = analyze_text(
            "Това е примерна българска новинарска статия."
        )

        expected_fields = {
            "word_count",
            "filtered_word_count",
            "character_count",
            "preview",
            "cleaned_text",
            "tokens",
            "risk_score",
            "risk_level",
            "risk_messages",
            "risk_details",
        }

        self.assertTrue(
            expected_fields.issubset(result.keys())
        )

    def test_analyzer_word_count(self):
        text = "Една кратка българска новина."

        result = analyze_text(text)

        self.assertEqual(result["word_count"], 4)
        self.assertEqual(
            result["character_count"],
            len(text),
        )

    def test_analyzer_rejects_non_string(self):
        with self.assertRaises(TypeError):
            analyze_text(123)


class FlaskApplicationTests(unittest.TestCase):
    """Функционални тестове за Flask приложението."""

    def setUp(self):
        self.original_testing = app.config.get("TESTING", False)
        self.original_login_disabled = app.config.get("LOGIN_DISABLED", False)

        app.config.update(
            TESTING=True,
            LOGIN_DISABLED=True,
        )

        self.client = app.test_client()

    def tearDown(self):
        app.config["TESTING"] = self.original_testing
        app.config["LOGIN_DISABLED"] = self.original_login_disabled

    def test_home_requires_login(self):
        app.config["LOGIN_DISABLED"] = False

        response = self.client.get("/")

        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

        app.config["LOGIN_DISABLED"] = True

    def test_home_page_loads(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            "Анализ на достоверността".encode("utf-8"),
            response.data,
        )

    def test_empty_form_submission(self):
        response = self.client.post(
            "/",
            data={"news_text": ""},
        )

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            "Моля, въведете текст".encode("utf-8"),
            response.data,
        )

    def test_suspicious_text_submission(self):
        suspicious_text = (
            "ШОКИРАЩО!!! СПЕШНО СПОДЕЛЕТЕ тази новина, "
            "преди да я изтрият!"
        )

        response = self.client.post(
            "/",
            data={"news_text": suspicious_text},
        )

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            "Оценка: висок".encode("utf-8"),
            response.data,
        )

        self.assertIn(
            "Точки: 56".encode("utf-8"),
            response.data,
        )

    def test_text_length_limit(self):
        response = self.client.post(
            "/",
            data={"news_text": "а" * 20001},
        )

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            "не трябва да надвишава".encode("utf-8"),
            response.data,
        )


if __name__ == "__main__":
    unittest.main()
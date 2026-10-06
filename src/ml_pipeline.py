import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import DATASET_FILE, MODELS_DIR


class FakeNewsDetector:
    """Обучава и използва модели за оценка на риск от фалшива новина."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )
        self.models = self._create_models()
        self.is_fitted = False

    @staticmethod
    def _create_models():
        return {
            "naive_bayes": MultinomialNB(),
            "svm": CalibratedClassifierCV(
                estimator=SVC(kernel="linear", random_state=42),
                method="sigmoid",
                cv=3,
            ),
            "neural_net": MLPClassifier(
                hidden_layer_sizes=(50,),
                max_iter=1000,
                random_state=42,
            ),
        }

    def train(self, csv_path=None):
        """Обучава моделите и записва измерените показатели в metrics.json."""
        data_path = Path(csv_path) if csv_path else DATASET_FILE

        if not data_path.exists():
            raise ValueError(f"Файлът с данни не е намерен: {data_path}")

        data = pd.read_csv(data_path)

        required_columns = {"title", "text", "label"}
        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Липсват задължителни колони: {missing}")

        data = data.dropna(subset=["title", "text", "label"]).copy()

        if data.empty:
            raise ValueError("Базата данни няма записи за обучение.")

        texts = (
            data["title"].astype(str).str.strip()
            + " "
            + data["text"].astype(str).str.strip()
        ).str.strip()

        label_map = {
            "real": 0,
            "reliable": 0,
            "true": 0,
            "0": 0,
            "fake": 1,
            "false": 1,
            "1": 1,
        }

        raw_labels = data["label"].astype(str).str.strip().str.lower()
        labels = raw_labels.map(label_map)

        valid_rows = labels.notna() & texts.ne("")
        texts = texts[valid_rows]
        labels = labels[valid_rows].astype(int)

        if labels.nunique() != 2:
            raise ValueError(
                "Трябват записи и с етикет real, и с етикет fake."
            )

        class_counts = labels.value_counts()

        if class_counts.min() < 10:
            raise ValueError(
                "За начало добави поне 10 real и 10 fake записа."
            )

        texts_train, texts_test, labels_train, labels_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            random_state=42,
            stratify=labels,
        )

        test_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )

        train_vectors = test_vectorizer.fit_transform(texts_train)
        test_vectors = test_vectorizer.transform(texts_test)

        evaluation_models = self._create_models()
        metrics = {}

        for model_name, model in evaluation_models.items():
            model.fit(train_vectors, labels_train)
            predictions = model.predict(test_vectors)

            metrics[model_name] = {
                "accuracy": round(float(accuracy_score(labels_test, predictions)), 4),
                "precision": round(
                    float(precision_score(labels_test, predictions, zero_division=0)),
                    4,
                ),
                "recall": round(
                    float(recall_score(labels_test, predictions, zero_division=0)),
                    4,
                ),
                "f1_score": round(
                    float(f1_score(labels_test, predictions, zero_division=0)),
                    4,
                ),
            }

        # След теста обучаваме финалните модели с всички налични записи.
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
        )

        all_vectors = self.vectorizer.fit_transform(texts)
        self.models = self._create_models()

        for model in self.models.values():
            model.fit(all_vectors, labels)

        self.is_fitted = True
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, MODELS_DIR / "vectorizer.joblib")

        for model_name, model in self.models.items():
            joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

        training_report = {
            "total_records": int(len(texts)),
            "real_records": int((labels == 0).sum()),
            "fake_records": int((labels == 1).sum()),
            "test_records": int(len(texts_test)),
            "metrics": metrics,
        }

        with (MODELS_DIR / "metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(training_report, file, ensure_ascii=False, indent=4)

        return training_report

    def load_models(self, models_dir=None):
        """Зарежда предварително обучените модели от папката models."""
        models_path = Path(models_dir) if models_dir else MODELS_DIR

        try:
            self.vectorizer = joblib.load(models_path / "vectorizer.joblib")
            self.models = {
                "naive_bayes": joblib.load(models_path / "naive_bayes.joblib"),
                "svm": joblib.load(models_path / "svm.joblib"),
                "neural_net": joblib.load(models_path / "neural_net.joblib"),
            }
            self.is_fitted = True
            return True
        except (FileNotFoundError, EOFError, ValueError):
            self.is_fitted = False
            return False

    def predict(self, text, model_name):
        """Връща [вероятност за реална, вероятност за фалшива новина]."""
        model = self.models.get(model_name)

        if not self.is_fitted or model is None or not text or not text.strip():
            return [0.5, 0.5]

        text_vector = self.vectorizer.transform([text])
        probabilities = list(model.predict_proba(text_vector)[0])
        classes = list(model.classes_)

        real_index = classes.index(0)
        fake_index = classes.index(1)

        return [
            float(probabilities[real_index]),
            float(probabilities[fake_index]),
        ]


if __name__ == "__main__":
    detector = FakeNewsDetector()

    try:
        report = detector.train()
        print("Моделите са обучени успешно.")
        print(f"Общо записи: {report['total_records']}")
        print(f"Реални новини: {report['real_records']}")
        print(f"Фалшиви новини: {report['fake_records']}")

        for model_name, values in report["metrics"].items():
            print(
                f"{model_name}: "
                f"accuracy={values['accuracy']:.2%}, "
                f"F1={values['f1_score']:.2%}"
            )
    except ValueError as error:
        print(f"Обучението не е стартирано: {error}")
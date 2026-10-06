from urllib.parse import urlparse
import ast
import base64
import csv
import datetime
import io
import json
import os
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from flask import Flask, abort, flash, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import DEBUG, HISTORY_FILE, SECRET_KEY, USERS_FILE
from src.analyzer import analyze_text


app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY

login_manager = LoginManager(app)
login_manager.login_view = "login"

HISTORY_COLUMNS = ["username", "text_preview", "predictions", "timestamp"]

TRUSTED_DOMAINS = {
    "btvnovinite.bg": "bTV Новините",
    "bntnews.bg": "БНТ Новини",
    "bta.bg": "Българска телеграфна агенция",
    "gov.bg": "Министерски съвет",
    "mon.bg": "Министерство на образованието и науката",
    "nsi.bg": "Национален статистически институт",
    "mvr.bg": "Министерство на вътрешните работи",
    "parliament.bg": "Народно събрание",
    "president.bg": "Президентство на Република България",
    "ec.europa.eu": "Европейска комисия",
}


def assess_source(url):
    """Проверява дали URL е от предварително разпознат надежден източник."""
    if not url:
        return {
            "status": "not_provided",
            "name": None,
            "message": "Не е въведен URL, затова източникът не може да бъде проверен."
        }

    try:
        domain = urlparse(url).netloc.lower().removeprefix("www.")

        for trusted_domain, source_name in TRUSTED_DOMAINS.items():
            if domain == trusted_domain or domain.endswith("." + trusted_domain):
                return {
                    "status": "trusted",
                    "name": source_name,
                    "message": f"Разпознат надежден източник: {source_name}."
                }

        return {
            "status": "unknown",
            "name": None,
            "message": "Източникът не е сред предварително разпознатите надеждни домейни."
        }

    except ValueError:
        return {
            "status": "unknown",
            "name": None,
            "message": "URL адресът не може да бъде проверен."
        }

class User(UserMixin):
    def __init__(self, username, role="user"):
        self.id = username
        self.username = username
        self.role = role


def get_user(username):
    """Връща данните на потребител по неговото име."""
    if not USERS_FILE.exists():
        return None

    with USERS_FILE.open(mode="r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("username") == username:
                return {
                    "username": username,
                    "password_hash": row.get("password_hash", ""),
                    "role": row.get("role", "user") or "user",
                }

    return None


@login_manager.user_loader
def load_user(user_id):
    user = get_user(user_id)

    if user is None:
        return None

    return User(user["username"], user["role"])


def user_exists(username):
    return get_user(username) is not None


def add_user(username, password):
    file_is_empty = not USERS_FILE.exists() or USERS_FILE.stat().st_size == 0

    with USERS_FILE.open(mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)

        if file_is_empty:
            writer.writerow(["username", "password_hash", "role"])

        writer.writerow(
            [
                username,
                generate_password_hash(password),
                "user",
            ]
        )


def verify_user(username, password):
    user = get_user(username)

    return bool(
        user
        and user["password_hash"]
        and check_password_hash(user["password_hash"], password)
    )


def ensure_history_file():
    """Създава коректен празен файл за история при нужда."""
    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        return

    with HISTORY_FILE.open(mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HISTORY_COLUMNS)


def parse_predictions(value):
    """Чете както новите JSON записи, така и старите записи в историята."""
    try:
        predictions = json.loads(value)
    except json.JSONDecodeError:
        try:
            predictions = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            predictions = {}

    return predictions if isinstance(predictions, dict) else {}


def scrape_news_text(url):
    """Извлича основния текст на статия от публичен URL адрес."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return None

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Премахваме елементи, които не са част от статията.
        for element in soup.select("script, style, nav, footer, aside, form, noscript"):
            element.decompose()

        # Търсим основното съдържание на статията.
        article = (
            soup.select_one("article")
            or soup.select_one('[itemprop="articleBody"]')
            or soup.select_one(".article-content")
            or soup.select_one(".article-body")
            or soup.select_one("main")
        )

        if article is None:
            return None

        paragraphs = article.find_all("p")

        text_parts = [
            paragraph.get_text(" ", strip=True)
            for paragraph in paragraphs
            if paragraph.get_text(strip=True)
        ]

        text = "\n".join(dict.fromkeys(text_parts))
        return text[:20_000] if text else None

    except requests.RequestException:
        return None

def generate_prediction_plot(predictions):
    """Създава графика на вероятностите за фалшива новина."""
    models = []
    fake_probabilities = []

    for model_name, prediction in predictions.items():
        if prediction and len(prediction) > 1:
            models.append(model_name)
            fake_probabilities.append(prediction[1])

    if not models:
        return None

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(models, fake_probabilities, color=["#ef4444", "#3b82f6", "#a855f7"])
    axis.set_title("Вероятност за фалшива новина по модели")
    axis.set_ylabel("Вероятност")
    axis.set_ylim(0, 1)

    image = io.BytesIO()
    figure.tight_layout()
    figure.savefig(image, format="png")
    plt.close(figure)

    image.seek(0)

    return (
        "data:image/png;base64,"
        + base64.b64encode(image.read()).decode("utf-8")
    )


ensure_history_file()


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            flash("Моля, попълнете всички полета.", "error")
        elif len(username) > 50 or not username.replace("_", "").replace("-", "").isalnum():
            flash(
                "Потребителското име може да съдържа само букви, цифри, тире и долна черта.",
                "error",
            )
        elif len(password) < 8:
            flash("Паролата трябва да съдържа поне 8 символа.", "error")
        elif password != confirm_password:
            flash("Паролите не съвпадат.", "error")
        elif user_exists(username):
            flash("Потребителското име вече съществува.", "error")
        else:
            add_user(username, password)
            flash("Регистрацията е успешна. Моля, влезте в системата.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Моля, попълнете всички полета.", "error")
        elif not verify_user(username, password):
            flash("Невалидно потребителско име или парола.", "error")
        else:
            user = load_user(username)
            login_user(user)
            flash("Успешно влизане.", "success")
            return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Успешно излизане.", "success")
    return redirect(url_for("login"))

@app.route("/history")
@login_required
def history():
    history_entries = []

    if HISTORY_FILE.exists():
        with HISTORY_FILE.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for raw_row in reader:
                row = {
                    str(key).strip(): value
                    for key, value in raw_row.items()
                    if key is not None
                }

                if row.get("username", "").strip() != current_user.id:
                    continue

                timestamp = row.get("timestamp", "").strip()

                history_entries.append({
                    "text": row.get("text_preview", ""),
                    "preds": parse_predictions(row.get("predictions", "")),
                    "time": timestamp,
                    "timestamp": timestamp,
                })

    return render_template("history.html", history_entries=history_entries)


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    news_text = ""
    result = None
    error = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        news_url = request.form.get("news_url", "").strip()

        if news_url and not news_text:
            news_text = scrape_news_text(news_url)

            if not news_text:
                error = "Неуспешно извличане на текст от предоставения URL."

        if not news_text and error is None:
            error = "Моля, въведете текст на новинарска статия или URL."
        elif news_text and len(news_text) > 20_000:
            error = "Текстът не трябва да надвишава 20 000 символа."
        elif news_text:
            try:
                result = analyze_text(news_text)
                result["source_assessment"] = assess_source(news_url)
                result["plot_url"] = generate_prediction_plot(
                    result["ml_predictions"]
                )

                with HISTORY_FILE.open(
                    "a",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            current_user.id,
                            news_text[:100] + "..."
                            if len(news_text) > 100
                            else news_text,
                            json.dumps(result["ml_predictions"]),
                            datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        ]
                    )

            except Exception:
                error = "Възникна проблем при анализа. Опитайте отново."

    return render_template(
        "index.html",
        news_text=news_text,
        result=result,
        error=error,
    )


@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    remaining_entries = []

    with HISTORY_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("username") != current_user.id:
                remaining_entries.append(row)

    with HISTORY_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(remaining_entries)

    return redirect(url_for("history"))


@app.route("/admin")
@login_required
def admin():
    if current_user.role != "admin":
        abort(403)

    users = []

    with USERS_FILE.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            users.append(
                {
                    "username": row.get("username", ""),
                    "role": row.get("role", "user") or "user",
                }
            )

    with HISTORY_FILE.open("r", encoding="utf-8", newline="") as file:
        total_news = sum(1 for _ in csv.DictReader(file))

    return render_template(
        "admin.html",
        users=users,
        total_users=len(users),
        total_news=total_news,
    )


if __name__ == "__main__":
    app.run(debug=DEBUG)
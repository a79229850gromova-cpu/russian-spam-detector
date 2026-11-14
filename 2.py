# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ============================================================
# 1. Загрузка старого датасета (СПАМ)
# ============================================================

SPAM_PATH = r"C:/Users/user/Downloads/processed_combined.parquet"
df_spam = pd.read_parquet(SPAM_PATH)
df_spam = df_spam[["message", "label"]].dropna()
df_spam.columns = ["text", "y"]
df_spam["y"] = df_spam["y"].astype(int)  # 0 = норма, 1 = спам


# ============================================================
# 2. Загрузка и очистка нового датасета (ФЕЙКИ)
# ============================================================

FAKE_PATH = r"C:/Users/user/Downloads/Dataset_Teachers.csv"

# Читаем со второй строки (header=1)
df_fake = pd.read_csv(FAKE_PATH, header=1)

# Убираем лишние пробелы из названий колонок
df_fake.columns = df_fake.columns.str.strip()

# Оставляем только нужные колонки и удаляем пустые строки
df_fake = df_fake[["текст повідомлень", "мітка", "Мова"]].dropna()

# Оставляем только русский язык (учитываем и 'Russian', и 'russian')
df_fake = df_fake[df_fake["Мова"].astype(str).str.strip().str.lower() == "russian"]

# Приводим метки к нижнему регистру и убираем пробелы
clean_labels = df_fake["мітка"].astype(str).str.strip().str.lower()

# Переводим слова в числа: 'фейк' -> 1, 'правда' -> 0
df_fake["y"] = clean_labels.map({"фейк": 1, "правда": 0})

# Оставляем только текст и числовую метку, отбрасываем всё, что не распозналось
df_fake = df_fake[["текст повідомлень", "y"]].dropna()
df_fake.columns = ["text", "y"]
df_fake["y"] = df_fake["y"].astype(int)

print("Загружено русских строк из датасета фейков:", len(df_fake))
print(df_fake["y"].value_counts())


# ============================================================
# 3. Балансировка и объединение
# ============================================================

# Берём 100 000 строк из спам-датасета
if len(df_spam) > 100_000:
    df_spam = df_spam.sample(n=100_000, random_state=42)

# Усиливаем вес датасета с фейками (повторяем его 40 раз),
# чтобы ~420 примеров фейков не потерялись на фоне 100 000 спам-сообщений
df_fake_upsampled = pd.concat([df_fake] * 40, ignore_index=True)

df = pd.concat([df_spam, df_fake_upsampled], ignore_index=True)
df["text"] = df["text"].astype(str)

# Перемешиваем
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nРазмер объединенного датасета:", df.shape)
print("\nРаспределение классов (0 = обычное, 1 = спам или фейк):")
print(df["y"].value_counts())


# ============================================================
# 4. Разделение, TF-IDF и обучение
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["y"],
    test_size=0.2,
    random_state=42,
    stratify=df["y"]
)

vectorizer = TfidfVectorizer(
    max_features=60_000,
    ngram_range=(1, 2),
    min_df=2,
    lowercase=True
)

X_train_v = vectorizer.fit_transform(X_train)
X_test_v = vectorizer.transform(X_test)

print("\nОбучение модели...")

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train_v, y_train)
print("Обучение завершено.")


# ============================================================
# 5. Сохранение и проверка
# ============================================================

joblib.dump(clf, "spam_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("\nФайлы spam_model.joblib и tfidf_vectorizer.joblib успешно обновлены!")

pred = clf.predict(X_test_v)
print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
print("\nОтчёт классификации:")
print(
    classification_report(
        y_test,
        pred,
        target_names=["обычное", "спам / фейк"]
    )
)
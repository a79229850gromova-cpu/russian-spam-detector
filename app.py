# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import numpy as np

# ============================================================
# НАСТРОЙКИ СТРАНИЦЫ
# ============================================================

st.set_page_config(
    page_title="АнтиФейк — проверка сообщений",
    page_icon="🛡️",
    layout="centered"
)

# Небольшой кастомный CSS для аккуратного вида
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ЗАГРУЗКА МОДЕЛИ (с кэшированием для быстрой работы)
# ============================================================

@st.cache_resource
def load_assets():
    mod = joblib.load("spam_model.joblib")
    vec = joblib.load("tfidf_vectorizer.joblib")
    return mod, vec


model, vectorizer = load_assets()

# ============================================================
# ШАПКА САЙТА
# ============================================================

st.title("🛡️ АнтиФейк Детектор")
st.caption(
    "Интеллектуальная система проверки текстов из мессенджеров и соцсетей "
    "на признаки дезинформации, вбросов и спама."
)

# Разделяем интерфейс на вкладки
tab_check, tab_info = st.tabs(["🔍 Проверка текста", "📊 О модели и метриках"])

# ============================================================
# ВКЛАДКА 1: ПРОВЕРКА СООБЩЕНИЯ
# ============================================================

with tab_check:
    # Инициализируем состояние для кнопок быстрых примеров
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    st.write("**Быстрые примеры для теста (нажмите, чтобы подставить):**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)

    with col_ex1:
        if st.button("🚨 Пример фейка", use_container_width=True):
            st.session_state.input_text = (
                "СРОЧНО! Перешлите всем знакомым! С завтрашнего дня вводят "
                "новые правила и начинают автоматически списывать деньги со всех карт, "
                "если не отправить отказ в специальном боте!"
            )

    with col_ex2:
        if st.button("📩 Пример спама", use_container_width=True):
            st.session_state.input_text = (
                "Поздравляем! Ваш номер выбран победителем розыгрыша! "
                "Заберите приз 500 000 рублей прямо сейчас по ссылке."
            )

    with col_ex3:
        if st.button("✅ Обычный текст", use_container_width=True):
            st.session_state.input_text = (
                "Привет! Напоминаю, что завтра встречаемся в 15:00 возле главного входа. "
                "Не забудь взять документы."
            )

    # Поле ввода текста
    message = st.text_area(
        "Текст сообщения для анализа:",
        value=st.session_state.input_text,
        placeholder="Вставьте сюда подозрительное сообщение или новость...",
        height=160
    )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        check_button = st.button("🔍 Проверить сообщение", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Очистить", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()

    # Логика проверки
    if check_button:
        if not message.strip():
            st.warning("⚠️ Сначала введите текст или выберите один из примеров выше.")
        else:
            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]
            probabilities = model.predict_proba(message_vector)[0]

            st.markdown("### Результат анализа")

            # Красивая карточка с рамкой
            with st.container(border=True):
                if prediction == 1:
                    confidence = probabilities[1]
                    st.error("🚨 **Обнаружены признаки ФЕЙКА или СПАМА**")

                    c1, c2 = st.columns(2)
                    c1.metric("Вероятность угрозы", f"{confidence * 100:.1f}%")
                    c2.metric("Статус", "Не рекомендуется доверять")
                    st.progress(float(confidence))

                    # Показываем слова, которые сильнее всего повлияли на решение модели
                    feature_names = np.array(vectorizer.get_feature_names_out())
                    coefs = model.coef_[0]
                    msg_indices = message_vector.nonzero()[1]

                    if len(msg_indices) > 0:
                        word_scores = [(feature_names[i], coefs[i] * message_vector[0, i]) for i in msg_indices]
                        # Берем топ-5 слов с положительным весом (в сторону фейка/спама)
                        suspicious_words = [w for w, s in sorted(word_scores, key=lambda x: x[1], reverse=True) if
                                            s > 0][:5]

                        if suspicious_words:
                            st.caption("🔎 **Ключевые слова-триггеры, на которые отреагировала модель:** " + ", ".join(
                                f"`{w}`" for w in suspicious_words))

                else:
                    confidence = probabilities[0]
                    st.success("✅ **Сообщение выглядит ДОСТОВЕРНЫМ (не фейк и не спам)**")

                    c1, c2 = st.columns(2)
                    c1.metric("Уверенность модели", f"{confidence * 100:.1f}%")
                    c2.metric("Статус", "Безопасно")
                    st.progress(float(confidence))

# ============================================================
# ВКЛАДКА 2: О МОДЕЛИ
# ============================================================

with tab_info:
    st.subheader("Архитектура и качество модели")

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", "98,83%")
        m2.metric("Recall (фейк/спам)", "96%")
        m3.metric("F1-score", "97%")

    st.markdown(
        """
        **Технические характеристики:**
        * **Алгоритм классификации:** Логистическая регрессия (`LogisticRegression` с балансировкой классов `class_weight="balanced"`).
        * **Векторизация текста:** TF-IDF (учитывает отдельные слова и устойчивые словосочетания — биграммы, словарь до 60 000 признаков).
        * **Обучающая выборка:** 116 680 русскоязычных сообщений (объединённый корпус спам-рассылок и верифицированных фейков из соцсетей и Telegram-каналов).
        * **Разделение выборки:** 80% обучение / 20% тестирование.
        """
    )
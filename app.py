import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="КиноПоиск ML Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0f1115;
        color: #e0e6ed;
    }
    .stApp {
        background-color: #0f1115;
    }
    h1, h2, h3 {
        color: #ff9f1c !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 159, 28, 0.2);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #ff9f1c;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #a0aec0;
        margin-top: 5px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff9f1c 0%, #ff5e36 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(255, 94, 54, 0.3) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 94, 54, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

st.title("🎬 Интеллектуальный классификатор фильмов")
st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🎯 Определение жанра",
    "📊 Аналитика & Статистика",
    "📖 Справка & API Документация"
])

with tab1:
    st.header("🔍 Прогнозирование категории фильма")
    st.write("Введите название и описание фильма (сюжет) на русском языке, и наша обученная модель мгновенно определит его основную категорию.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Данные фильма")
        
        movie_title = st.text_input(
            "Название фильма",
            placeholder="Например: Побег из Шоушенка",
            help="Необязательное поле для персонализации ответа."
        )
        
        sample_plot = (
            "Бухгалтер Энди Дюфрейн обвинён в убийстве собственной жены и её любовника. "
            "Оказавшись в тюрьме под названием Шоушенк, он сталкивается с жестокостью и беззаконием, "
            "но благодаря своему уму, терпению и доброте находит друзей среди заключенных и разрабатывает "
            "невероятный план побега к свободе."
        )
        
        use_sample = st.checkbox("Заполнить демо-примером сюжета (драма)")
        
        plot_desc = st.text_area(
            "Сюжет / Описание фильма",
            value=sample_plot if use_sample else "",
            height=200,
            placeholder="Введите описание сюжета фильма на русском...",
            help="Чем подробнее описание сюжета, тем выше точность определения категории!"
        )
        
        predict_btn = st.button("🚀 Определить жанр")
        
    with col2:
        st.subheader("🔮 Результат классификации")
        
        if predict_btn:
            if not plot_desc.strip():
                st.warning("Пожалуйста, заполните описание сюжета фильма перед запуском!")
            else:
                with st.spinner("Анализируем сюжет моделированием..."):
                    try:
                        payload = {"title": movie_title, "description": plot_desc}
                        r = requests.post(f"{API_URL}/predict", json=payload)
                        
                        if r.status_code == 200:
                            data = r.json()
                            genre = data["predicted_genre"]
                            confidence = data["confidence"]
                            
                            st.success("Анализ завершен успешно!")
                            
                            st.markdown(f"""
                            <div style="background: rgba(255,159,28,0.08); padding: 30px; border-radius: 12px; border: 1px solid #ff9f1c; margin-top: 15px;">
                                <h3 style="margin: 0; color: #a0aec0; font-size: 1.1rem;">Прогнозируемый жанр для {data['title']}:</h3>
                                <div style="font-size: 3rem; font-weight: 800; color: #ff9f1c; margin: 10px 0;">{genre}</div>
                                <div style="margin-top: 15px;">
                                    <span style="font-size: 0.95rem; color: #e0e6ed;">Уверенность модели: <b>{confidence*100:.1f}%</b></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.progress(confidence)
                            
                        else:
                            st.error(f"Не удалось связаться с сервером предсказаний. Ошибка: {r.text}")
                    except Exception as e:
                        st.error(f"Ошибка соединения с API сервером. Пожалуйста, убедитесь, что API сервер запущен на порту 8000! Детали: {e}")
        else:
            st.info("Введите описание сюжета в левой колонке и нажмите кнопку запуска предсказания.")

with tab2:
    st.header("📊 Статистика и анализ данных")
    st.write("Сводная интерактивная статистика по 250 лучшим фильмам КиноПоиска.")
    
    stats_data = None
    try:
        r = requests.get(f"{API_URL}/stats")
        if r.status_code == 200:
            stats_data = r.json()
    except Exception as e:
        workspace_dir = None
        for name in os.listdir(r"C:\Users\User\anacond"):
            if name.startswith("3"):
                workspace_dir = os.path.join(r"C:\Users\User\anacond", name)
                break
        if workspace_dir:
            csv_path = os.path.join(workspace_dir, "kinopoisk_top250.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['description'] = df['description'].fillna('').astype(str)
                df['genre'] = df['genre'].fillna('').astype(str)
                
                char_lengths = df['description'].apply(len)
                word_lengths = df['description'].apply(lambda x: len(x.split()))
                primary_genres = df['genre'].apply(lambda x: x.split(',')[0].strip().lower())
                
                stats_data = {
                    "total_movies": len(df),
                    "avg_description_chars": float(char_lengths.mean()),
                    "avg_description_words": float(word_lengths.mean()),
                    "genre_counts": primary_genres.value_counts().to_dict(),
                    "genre_shares_percent": (primary_genres.value_counts(normalize=True) * 100).round(2).to_dict(),
                    "ratings": {
                        "mean": float(df['rating'].mean()),
                        "min": float(df['rating'].min()),
                        "max": float(df['rating'].max())
                    },
                    "years": df['year'].value_counts().sort_index().to_dict()
                }

    if stats_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats_data['total_movies']}</div>
                <div class="metric-lbl">Фильмов в датасете</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{int(stats_data['avg_description_words'])}</div>
                <div class="metric-lbl">Средняя длина сюжета (слов)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{int(stats_data['avg_description_chars'])}</div>
                <div class="metric-lbl">Средняя длина сюжета (симв.)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{stats_data['ratings']['mean']:.2f}</div>
                <div class="metric-lbl">Средний рейтинг фильмов</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1], gap="large")
        
        with col_left:
            st.subheader("🍕 Распределение основных жанров фильмов")
            
            genre_df = pd.DataFrame({
                "Жанр": [k.capitalize() for k in stats_data['genre_counts'].keys()],
                "Количество": list(stats_data['genre_counts'].values())
            })
            
            fig_pie = px.pie(
                genre_df, 
                values="Количество", 
                names="Жанр", 
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.YlOrBr_r
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.subheader("📈 Выпуск фильмов по годам в топ-250")
            
            years_df = pd.DataFrame({
                "Год": list(stats_data['years'].keys()),
                "Фильмов": list(stats_data['years'].values())
            })
            
            fig_area = px.area(
                years_df, 
                x="Год", 
                y="Фильмов",
                color_discrete_sequence=['#ff9f1c']
            )
            fig_area.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e6ed'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
        st.subheader("🎯 Зависимость рейтинга фильма от длины описания сюжета")
        
        workspace_dir = None
        for name in os.listdir(r"C:\Users\User\anacond"):
            if name.startswith("3"):
                workspace_dir = os.path.join(r"C:\Users\User\anacond", name)
                break
        if workspace_dir:
            csv_path = os.path.join(workspace_dir, "kinopoisk_top250.csv")
            if os.path.exists(csv_path):
                df_scatter = pd.read_csv(csv_path)
                df_scatter['Длина описания (слов)'] = df_scatter['description'].fillna('').apply(lambda x: len(str(x).split()))
                df_scatter['Основной жанр'] = df_scatter['genre'].fillna('драма').apply(lambda x: x.split(',')[0].strip().capitalize())
                
                fig_scatter = px.scatter(
                    df_scatter,
                    x="Длина описания (слов)",
                    y="rating",
                    color="Основной жанр",
                    hover_name="title",
                    labels={"rating": "Рейтинг фильма"},
                    color_discrete_sequence=px.colors.qualitative.Antique
                )
                fig_scatter.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e6ed'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.error("Не удалось получить статистические данные. Пожалуйста, запустите FastAPI сервер!")

with tab3:
    st.header("📖 Руководство пользователя и справка по API")
    
    st.subheader("💡 Описание имеющихся команд и параметров")
    st.markdown("""
    Для взаимодействия с классификатором фильмов разработаны следующие программные интерфейсы (endpoints):
    
    1. **`POST /predict`** — Определение жанра фильма по описанию сюжета.
       - **Параметры тела запроса (JSON)**:
         - `title` *(строка, необязательно)*: название фильма для отображения в результатах.
         - `description` *(строка, обязательно)*: текст описания сюжета на русском языке.
       - **Формат ответа (JSON)**:
         ```json
         {
           "title": "Зеленая миля",
           "predicted_genre": "Драма",
           "confidence": 0.894
         }
         ```
         
    2. **`GET /stats`** — Сводная статистика по фильмам.
       - **Параметры**: нет.
       - **Формат ответа (JSON)**: содержит общее число фильмов, доли категорий в процентах, распределение по годам, характеристики длин текстов и средний рейтинг.
       
    3. **`GET /help`** — Возвращает это справочное руководство.
    """)
    
    st.subheader("💻 Интеграционные примеры кода")
    
    st.markdown("##### Запрос с использованием `cURL` в командной строке:")
    st.code("""
curl -X POST "http://127.0.0.1:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"title": "Форрест Гамп", "description": "Форрест Гамп - открытый и добрый парень с умственными ограничениями рассказывает свою историю."}'
    """, language="bash")
    
    st.markdown("##### Запрос с использованием библиотеки `requests` на Python:")
    st.code("""
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "title": "Интерстеллар",
    "description": "Когда засуха приводит человечество к продовольственному кризису, коллектив исследователей отправляется в путешествие сквозь червоточину."
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    data = response.json()
    print(f"Фильм: {data['title']}")
    print(f"Жанр: {data['predicted_genre']}")
    print(f"Уверенность: {data['confidence'] * 100:.2f}%")
    """, language="python")

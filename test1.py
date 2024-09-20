import streamlit as st
import sqlite3
import hashlib
import os
import openai
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from zipfile import ZipFile
import numpy as np

# Загрузка переменных окружения
load_dotenv()

# Настройка API ключа OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Настройка страницы
st.set_page_config(page_title="Creditworthiness App", layout="wide")

# Подключение к SQLite базе данных
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('PRAGMA foreign_keys = ON')

# Функция для создания таблицы пользователей
def create_tables():
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        name TEXT NOT NULL
    )
    ''')
    conn.commit()

create_tables()

# Функции для управления пользователями
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_username_exists(username):
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return True
    return False

def register_user(username, password, name):
    if check_username_exists(username):
        st.error("Имя пользователя уже существует. Пожалуйста, выберите другое имя.")
    else:
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password, name) VALUES (?, ?, ?)',
                  (username, hashed_password, name))
        conn.commit()
        st.success("Вы успешно создали аккаунт. Теперь вы можете войти в систему.")

def authenticate_user(username, password):
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(user[1], password):
        return user[0]  # Возвращаем user_id
    return None

def verify_password(stored_password, provided_password):
    return stored_password == hashlib.sha256(provided_password.encode()).hexdigest()

def check_authentication():
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        st.warning("Пожалуйста, войдите в систему для доступа к этой странице.")
        st.stop()

def logout():
    if "authenticated" in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.success("Вы вышли из системы.")
        st.experimental_rerun()

# Инициализация состояния сессии
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Навигация
st.sidebar.title("Навигация")

if "authenticated" in st.session_state and st.session_state.authenticated:
    if st.sidebar.button("Выйти"):
        logout()
    page = st.sidebar.radio("", ["Главная", "Чат бот", "Анализ кредитоспособности"])
else:
    page = st.sidebar.radio("", ["Главная", "Войти", "Регистрация"])

# ----------------- Главная страница -----------------
if page == "Главная":
    st.subheader("Приложение для анализа кредитоспособности")
    st.write("""
        Это приложение позволяет пользователям оценить свою кредитоспособность и пообщаться с чат-ботом.
        Пожалуйста, используйте меню навигации для регистрации или входа в систему.
    """)

# ----------------- Страница входа -----------------
elif page == "Войти":
    st.subheader("Вход")
    username = st.text_input("Имя пользователя")
    password = st.text_input("Пароль", type='password')
    if st.button("Войти"):
        user_id = authenticate_user(username, password)
        if user_id:
            st.success(f"Добро пожаловать, {username}")
            st.session_state.username = username
            st.session_state.user_id = user_id
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Неправильное имя пользователя или пароль")

# ----------------- Страница регистрации -----------------
elif page == "Регистрация":
    st.subheader("Создать новый аккаунт")
    new_username = st.text_input("Имя пользователя", key="register_username")
    new_password = st.text_input("Пароль", type='password', key="register_password")
    new_name = st.text_input("Ваше имя", key="register_name")
    if st.button("Зарегистрироваться", key="register_button"):
        if new_username and new_password and new_name:
            register_user(new_username, new_password, new_name)
        else:
            st.error("Пожалуйста, заполните все поля.")

# ----------------- Страница Чат бота -----------------
elif page == "Чат бот":
    check_authentication()

    def chatbot():
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": "Вы - дружелюбный помощник, готовый ответить на вопросы пользователя."}
            ]

        st.write("### Чат бот")
        user_input = st.text_input("Ваш вопрос:")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=st.session_state.messages
                )
                answer = response.choices[0].message['content']
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except openai.error.OpenAIError as e:
                st.error(f"Ошибка API OpenAI: {e}")
                st.write(e.__dict__)

        # Отображение диалога
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.write(f"**Вы:** {message['content']}")
            elif message["role"] == "assistant":
                st.write(f"**Чат бот:** {message['content']}")

    chatbot()

# ----------------- Страница Анализа кредитоспособности -----------------
elif page == "Анализ кредитоспособности":
    check_authentication()

    def creditworthiness_analysis():
        from zipfile import ZipFile

        # Словарь перевода кредитов
        loan_translation = {
            'Автокредит': 'Auto Loan',
            'Кредит на создание кредита': 'Credit-Builder Loan',
            'Персональный кредит': 'Personal Loan',
            'Кредит под залог жилья': 'Home Equity Loan',
            'Ипотечный кредит': 'Mortgage Loan',
            'Студенческий кредит': 'Student Loan',
            'Кредит на консолидацию долга': 'Debt Consolidation Loan',
            'Кредит до зарплаты': 'Payday Loan'
        }

        # Названия кредитов на русском для UI
        loan_names_russian = list(loan_translation.keys())

        # Функция для загрузки моделей из ZIP-файлов
        @st.cache(allow_output_mutation=True)
        def unzip_load(name):
            path = os.path.dirname(__file__)
            folder_path = os.path.join(path, r'C:\Users\Probook\pythonProjectcrusader\models')
            path_zip = os.path.join(folder_path, name + '.zip')
            with ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
            path_obj = os.path.join(folder_path, name + '.obj')
            return pickle.load(open(path_obj, 'rb'))

        # Загрузка моделей и скейлера
        scaler = unzip_load('scaler')
        model = unzip_load('model')

        # Значения по умолчанию для полей ввода
        age_default = 18
        annual_income_default = 15000.00
        accounts_default = 0
        credit_cards_default = 10
        delayed_payments_default = 5
        credit_card_ratio_default = 43.00
        emi_monthly_default = 0.00
        credit_history_default = 4
        loans_default = ['Студенческий кредит']
        missed_payment_default = 0
        minimum_payment_default = 0

        # Заголовок и описание
        st.title('Анализ кредитоспособности')
        st.caption('Made by Toleubek Erkenzhe')

        st.markdown('''
            Сервис, предназначенный для информации, с помощью которого вы можете оценить свою кредитоспособность за секунды и узнать возможность получения того или иного займа.
        ''')

        # Поля ввода
        with st.sidebar:
            st.header('Форма для кредитного скоринга')
            age = st.slider('Сколько вам лет?', min_value=18, max_value=100, step=1, value=age_default)
            annual_income = st.number_input('Какой ваш годовой доход?', min_value=0.00, max_value=300000.00, value=annual_income_default)
            accounts = st.number_input('Сколько у вас банковских счетов?', min_value=0, max_value=20, step=1, value=accounts_default)
            credit_cards = st.number_input('Сколько у вас кредитных карт?', min_value=0, max_value=12, step=1, value=credit_cards_default)
            delayed_payments = st.number_input('Сколько у вас просроченных платежей?', min_value=0, max_value=20, step=1, value=delayed_payments_default)
            credit_card_ratio = st.slider('Каков коэффициент использования вашей кредитной карты?', min_value=0.00, max_value=100.00, value=credit_card_ratio_default)
            emi_monthly = st.number_input('Какую сумму ежемесячного платежа вы платите?', min_value=0.00, max_value=5000.00, value=emi_monthly_default)
            credit_history = st.number_input('Сколько месяцев вашей кредитной истории?', min_value=0, max_value=500, step=1, value=credit_history_default)

            loans_russian = st.multiselect('Какие кредиты у вас есть?', loan_names_russian, default=loans_default)
            loans_english = [loan_translation[loan] for loan in loans_russian]

            missed_payment = st.radio('Были ли у вас пропуски в платежах за последние 12 месяцев?', ['Да', 'Нет'], index=missed_payment_default)

            run = st.button('Запустить скоринг!')

        st.header('Результаты кредитного скоринга')

        col1, col2 = st.columns([3, 2])

        with col2:
            x1 = [0, 6, 0]
            x2 = [0, 4, 0]
            x3 = [0, 2, 0]
            y = ['0', '1', '2']

            f, ax = plt.subplots(figsize=(5,2))

            p1 = sns.barplot(x=x1, y=y, color='#3EC300')
            p1.set(xticklabels=[], yticklabels=[])
            p1.tick_params(bottom=False, left=False)
            p2 = sns.barplot(x=x2, y=y, color='#FAA300')
            p2.set(xticklabels=[], yticklabels=[])
            p2.tick_params(bottom=False, left=False)
            p3 = sns.barplot(x=x3, y=y, color='#FF331F')
            p3.set(xticklabels=[], yticklabels=[])
            p3.tick_params(bottom=False, left=False)

            plt.text(0.7, 1.05, "Плохой", horizontalalignment='left', size='medium', color='white', weight='semibold')
            plt.text(2.5, 1.05, "Обычный", horizontalalignment='left', size='medium', color='white', weight='semibold')
            plt.text(4.7, 1.05, "Хороший", horizontalalignment='left', size='medium', color='white', weight='semibold')

            ax.set(xlim=(0, 6))
            sns.despine(left=True, bottom=True)

            figure = st.pyplot(f)

        with col1:
            placeholder = st.empty()

            if run:
                def transform_resp(resp):
                    data = {
                        'Age': resp['age'],
                        'Annual_Income': resp['annual_income'],
                        'Num_Bank_Accounts': resp['accounts'],
                        'Num_Credit_Card': resp['credit_cards'],
                        'Num_of_Delayed_Payment': resp['delayed_payments'],
                        'Credit_Utilization_Ratio': resp['credit_card_ratio'],
                        'Total_EMI_per_month': resp['emi_monthly'],
                        'Credit_History_Age_Formated': resp['credit_history'],
                        'Missed_Payment_Day': 1 if resp['missed_payment'] == 'Да' else 0,
                        'Payment_of_Min_Amount_Yes': resp['minimum_payment'],
                        'Auto_Loan': 1 if 'Auto Loan' in resp['loans'] else 0,
                        'Credit-Builder_Loan': 1 if 'Credit-Builder Loan' in resp['loans'] else 0,
                        'Personal_Loan': 1 if 'Personal Loan' in resp['loans'] else 0,
                        'Home_Equity_Loan': 1 if 'Home Equity Loan' in resp['loans'] else 0,
                        'Mortgage_Loan': 1 if 'Mortgage Loan' in resp['loans'] else 0,
                        'Student_Loan': 1 if 'Student Loan' in resp['loans'] else 0,
                        'Debt_Consolidation_Loan': 1 if 'Debt Consolidation Loan' in resp['loans'] else 0,
                        'Payday_Loan': 1 if 'Payday Loan' in resp['loans'] else 0
                    }
                    return pd.DataFrame([data])

                resp = {
                    'age': age,
                    'annual_income': annual_income,
                    'accounts': accounts,
                    'credit_cards': credit_cards,
                    'delayed_payments': delayed_payments,
                    'credit_card_ratio': credit_card_ratio,
                    'emi_monthly': emi_monthly,
                    'credit_history': credit_history,
                    'loans': loans_english,
                    'missed_payment': missed_payment,
                    'minimum_payment': minimum_payment_default
                }

                output = transform_resp(resp)
                output.loc[:, :] = scaler.transform(output)

                credit_score = model.predict(output)[0]

                if credit_score == 1:
                    st.balloons()
                    placeholder.markdown('Ваш кредитный рейтинг **ХОРОШИЙ**! Поздравляем!')
                    st.markdown('Этот рейтинг показывает, что вероятность погашения кредита высокая, так что риск минимален.')
                    # Добавляем стрелку на график
                    t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
                elif credit_score == 0:
                    placeholder.markdown('Ваш кредитный рейтинг **ОБЫЧНЫЙ**.')
                    st.markdown('Этот рейтинг показывает, что кредит скорее всего будет погашен, но возможны пропуски платежей.')
                    t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
                elif credit_score == -1:
                    placeholder.markdown('Ваш кредитный рейтинг **ПЛОХОЙ**.')
                    st.markdown('Этот рейтинг показывает, что вероятность непогашения кредита высокая, так что риск велик.')
                    t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')

                # Добавляем стрелку на график
                ax.add_patch(t1)
                figure.pyplot(f)

                # График вероятностей
                prob_fig, ax = plt.subplots()
                with st.expander('Нажмите, чтобы увидеть, насколько алгоритм был уверен'):
                    plt.pie(model.predict_proba(output)[0], labels=['Плохой', 'Обычный', 'Хороший'], autopct='%.0f%%')
                    st.pyplot(prob_fig)

                # Важность признаков
                with st.expander('Нажмите, чтобы увидеть, насколько каждая характеристика влияет'):
                    importance = model.feature_importances_
                    importance = pd.DataFrame(importance)
                    columns = pd.DataFrame(['Age', 'Annual_Income', 'Num_Bank_Accounts',
                                            'Num_Credit_Card', 'Num_of_Delayed_Payment',
                                            'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                                            'Credit_History_Age_Formated', 'Auto_Loan',
                                            'Credit-Builder_Loan', 'Personal_Loan', 'Home_Equity_Loan',
                                            'Mortgage_Loan', 'Student_Loan', 'Debt_Consolidation_Loan',
                                            'Payday_Loan', 'Missed_Payment_Day', 'Payment_of_Min_Amount_Yes'])

                    importance = pd.concat([importance, columns], axis=1)
                    importance.columns = ['importance', 'index']
                    importance_fig = round(importance.set_index('index') * 100.00, 2)
                    loans = ['Auto_Loan', 'Credit-Builder_Loan', 'Personal_Loan',
                             'Home_Equity_Loan', 'Mortgage_Loan', 'Student_Loan',
                             'Debt_Consolidation_Loan', 'Payday_Loan']

                    # Суммирование кредитов
                    Loans = importance_fig.loc[loans].sum().reset_index()
                    Loans['index'] = 'Loans'
                    Loans.columns = ['index', 'importance']
                    importance_fig = importance_fig.drop(loans, axis=0).reset_index()
                    importance_fig = pd.concat([importance_fig, Loans], axis=0)
                    importance_fig.sort_values(by='importance', ascending=True, inplace=True)

                    # Построение графика
                    importance_figure, ax = plt.subplots()
                    bars = ax.barh('index', 'importance', data=importance_fig)
                    ax.bar_label(bars)
                    plt.ylabel('')
                    plt.xlabel('')
                    plt.xlim(0, 20)
                    sns.despine(right=True, top=True)
                    st.pyplot(importance_figure)

    creditworthiness_analysis()

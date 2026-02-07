from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
f = open('spam.txt').readlines()
a = open('not_spam.txt').readlines()
spam = []
not_spam = []
for i in f:
    if i != '':
        spam.append(i[:-2])
for i in a:
    not_spam.append(i[:-2])
print(not_spam)

print(spam)
#тренировочные данные
train_list = spam + not_spam

train_answ = [1]*len(spam) + [0]*len(not_spam)

print(f"\nДлины после объединения:")
print(f"train_list: {len(train_list)}")
print(f"train_answ: {len(train_answ)}")


spam_count = sum(train_answ)
norm_count = len(train_answ) - spam_count
print(f"\nБаланс классов:")
print(f"Спам сообщений: {spam_count}")
print(f"Нормальных сообщений: {norm_count}")

# Векторизация
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_list)
y = train_answ

print(f"\nВекторизовано: {X.shape[0]} примеров, {X.shape[1]} признаков")

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nРазделение данных:")
print(f"Тренировочные: {X_train.shape[0]} примеров")
print(f"Тестовые: {X_test.shape[0]} примеров")

# Обучение
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['Нормальные', 'Спам']))




class SpamModel:
    def __init__(self):
        self.model = model  # ваша обученная модель
        self.vectorizer = vectorizer  # ваш обученный векторайзер

    def predict(self, text):
        # преобразуем текст в вектор
        if isinstance(text, str):
            text = [text]

        text_vectorized = self.vectorizer.transform(text)

        # получаем предсказание
        prediction = self.model.predict(text_vectorized)

        # получаем вероятности
        probabilities = self.model.predict_proba(text_vectorized)

        return prediction[0], {
            'spam_probability': probabilities[0][1],
            'normal_probability': probabilities[0][0]
        }

    def predict_simple(self, text):
        """Упрощенный метод, возвращает только True/False"""
        pred, prob = self.predict(text)
        return pred == 1, prob


# создаем экземпляр для импорта
spam_model = SpamModel()



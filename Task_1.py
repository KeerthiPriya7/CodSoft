import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import string

path = "C:\\Users\\keert\\Downloads\\train_data.txt.zip"
data = pd.read_csv(path, sep=":::", names=["TITLE", "GENRE", "DESCRIPTION"], engine="python")

print(data.info())
print(data.describe())
print(data.isnull().sum())

plt.figure(figsize=(10, 10))
sns.countplot(data=data, y="GENRE", order=data["GENRE"].value_counts().index, hue="GENRE", palette="deep", legend=False)
plt.show()

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
print("First 10 stop words:", list(stop_words)[:10])

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z+]', ' ', text)
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["TextCleaning"] = data["DESCRIPTION"].apply(cleaning_data)
print(data)

small_sample = data.sample(n=5000, random_state=42)

X = small_sample["TextCleaning"]
y = small_sample["GENRE"]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)  
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = SVC()
model.fit(X_train_tfidf, Y_train)

accuracy = model.score(X_train_tfidf, Y_train)
print(f"Model Score on Training Set: {accuracy * 100:.2f}%")

Y_pred = model.predict(X_test_tfidf)

test_accuracy = model.score(X_test_tfidf, Y_test)
print(f"Model Score on Test Set: {test_accuracy * 100:.2f}%")

predictions_df = pd.DataFrame({"Actual Genre": Y_test, "Predicted Genre": Y_pred})
print(predictions_df.head(10))

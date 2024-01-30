import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

docs = pd.read_csv('C:\\Users\\keert\\OneDrive\\Desktop\\spam.csv', encoding='latin1')
X = docs['v2']
y = docs['v1']
print(docs.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2%}".format(accuracy))

print(classification_report(y_test, y_pred))

new_messages = ["Free entry to a concert!", "Meeting at 3 pm tomorrow", "Get rich quick!"]
new_messages_tfidf = vectorizer.transform(new_messages)
new_predictions = clf.predict(new_messages_tfidf)
for message, prediction in zip(new_messages, new_predictions):
    print(f"Message: {message}\nPrediction: {prediction}\n")

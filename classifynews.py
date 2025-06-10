# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('bbc_news_dataset.csv')

# %%
df.isna().sum()

# %%
X = df['Text']
y = df['Category']

# %%
category_names = df['Category'].unique()
print(category_names)

# %%
import seaborn as sns
plt.figure(figsize=(8,4))
sns.countplot(data=df, x = 'Category', order=df['Category'].value_counts().index)
plt.xticks(rotation = 45)
plt.title("News Categories Distribution")
plt.show()


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.head()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,ConfusionMatrixDisplay

# %%
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
print(f'Accuracy is {accuracy_score(y_test, y_preds)*100}')
print('\n', classification_report(y_test, y_preds))

# %%
cm = confusion_matrix(y_test, y_preds)
cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['business', 'tech', 'politics','sport','entertainment'])
cm_plot.plot()

# %%
news = [
    'The largest football match ended in a draw , with great performance from both '
    'The government announced new aid to Ukraine following catastrophic talks with '
]
news_preds = model.predict(news)
for text, prediction in zip(news,news_preds):
    print(f'Text = {text}\nPredicted Category = {prediction}')



import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def textprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    stopWords = stopwords.words('english')
    stopWords.append('quot')
    text = word_tokenize(text.lower())
    text = ' '.join([lemmatizer.lemmatize(word) for word in text if not word in set(stopWords)])
    return text

df = pd.read_csv('Suicide_Ideation_Dataset(Twitter-based).csv')
df = df.dropna()

df['Tweet'] = df['Tweet'].apply(textprocess)

df['Suicide'] = pd.get_dummies(df['Suicide']).drop(columns='Not Suicide post')

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Tweet'])
X = X.toarray()
Y = pd.Series.to_numpy(df['Suicide'])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model2 = XGBClassifier()

model2.fit(X_train,Y_train)

output = model2.predict(X_test)

score = accuracy_score(Y_test,output)


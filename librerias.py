import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def useLib():
    data = pd.read_csv('entrenamiento.csv')
    data['spam'] = np.where(data['spam']=='spam',1, 0)

    X_train, X_test, y_train, y_test = train_test_split(data['message'], 
                                                        data['spam'], 
                                                        test_size=0.3,
                                                        random_state=1234)

    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_train_vectorized.toarray().shape
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vectorizer.transform(X_test))
    
    print("Accuracy: \n", accuracy_score(y_test,predictions))
    print("Confusion Matrix: \n", confusion_matrix(y_test,predictions))

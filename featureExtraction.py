from cgitb import reset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extraccionCaracteristicas(df):
    
    #LabelEncoder
    le = LabelEncoder()

    # Normalizar Topic y Source
    df['Topic'] =  le.fit_transform(df['Topic'])
    df['Source'] = le.fit_transform(df['Source'])

    #TF-IDF  

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df['Text'])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()

    res = pd.DataFrame(denselist, columns=feature_names)
    res.to_csv('./export/tf-idf.csv', header=True, index=True)

    return df
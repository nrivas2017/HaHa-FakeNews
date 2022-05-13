from sklearn.preprocessing import LabelEncoder


def extraccionCaracteristicas(df):
    
    #LabelEncoder
    le = LabelEncoder()

    # Normalizar Topic y Source
    df['Topic'] =  le.fit_transform(df['Topic'])
    df['Source'] = le.fit_transform(df['Source'])

    #TF-IDF  (?)

    return df


    
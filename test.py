import pandas as pd
import nltk 
import string

#Instalar en la primera ejecucion
#nltk.download('popular')

df_inicial = pd.read_excel("./corpus/train.xlsx")


#Corpus en español de palabras vacias
stopword_es = nltk.corpus.stopwords.words('spanish')

#Funcion para eliminar palabras vacias y signos de puntuacion de Texto y Titular
def removeStopWords(df):
    aCols = ['Text','Headline']
    for sCol in aCols: 
        df[sCol] = df[sCol].str.lower()
        df[sCol] = df[sCol].apply(lambda x: ' '.join([palabra for palabra in x.split() if palabra not in (stopword_es)]))
        df[sCol] = df[sCol].str.translate(str.maketrans('', '', string.punctuation))

    return df

#Funcion para convertir columnas a cadenas y concatenarlas en una unica serie de datos
def mergeColumnas(df):
    nF = 0
    for sCol in list(df.columns[1:]):
        if nF == 0 : df['ColUnica'] = df[sCol].astype(str) + ' '
        else       : df['ColUnica'] = df['ColUnica'] + df[sCol].astype(str) + ' '
        nF=1
    df = df['ColUnica']

    return df


#Casos Preprocesamiento
#--------------------------------------------------------

# CASO 1: Merge columnas quitando palabras vacias
df_merge1 = removeStopWords(df_inicial)
df_merge1 = mergeColumnas(df_merge1)

#Prueba con exportacion a csv
df_merge1.to_csv('./export/pruebas/ej_merge1.csv')

# CASO 2: Merge columnas sin quitar palabras vacias ni signos de puntuacion
print(df_inicial)
df_merge2 = mergeColumnas(df_inicial)
df_merge2.to_csv('./export/pruebas/ej_merge2.csv')






    


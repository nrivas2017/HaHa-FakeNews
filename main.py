import pandas as pd, nltk, string, plotData

#Instalar en la primera ejecucion
#nltk.download('popular')

#Corpus en español de palabras vacias
stopword_es = nltk.corpus.stopwords.words('spanish')

#Funcion para eliminar palabras vacias y signos de puntuacion de Texto y Titular
def removeStopWords(df,n):
    if n == 0: aCols = ['Text','Headline']
    else     : aCols = ['Text']

    for sCol in aCols: 
        df[sCol] = df[sCol].str.lower()
        df[sCol] = df[sCol].apply(lambda x: ' '.join([palabra for palabra in x.split() if palabra not in (stopword_es)]))
        df[sCol] = df[sCol].str.translate(str.maketrans('', '', string.punctuation))


#Funcion para convertir columnas a cadenas y concatenarlas en una unica serie de datos
def mergeColumnas(df):
    nF = 0
    for sCol in list(df.columns[1:]):
        if nF == 0 : df['ColUnica'] = df[sCol].astype(str) + ' '
        else       : df['ColUnica'] = df['ColUnica'] + df[sCol].astype(str) + ' '
        nF=1

#Casos Preprocesamiento
#--------------------------------------------------------

df_inicial = pd.read_excel("./corpus/train.xlsx")
pltData = plotData.plotData(df_inicial)

#Se copian los dataframes para tomar cada caso
df_merge1 = df_inicial.copy() ; df_texto1 = (df_inicial['Text'].copy()).to_frame()
df_merge2 = df_inicial.copy() ; 

# CASO 1: Merge columnas quitando palabras vacias
removeStopWords(df_merge1,0) ; mergeColumnas(df_merge1)
df_merge1 = df_merge1['ColUnica']

#CASO 2: Merge columnas sin quitar palabras vacias
mergeColumnas(df_merge2)
df_merge2 = df_merge2['ColUnica']

#CASO 3: Solo Text quitando palabras vacias
removeStopWords(df_texto1,1)

#CASO 4: Solo Text sin quitar palabras vacias
df_texto2 = (df_inicial['Text'].copy()).to_frame()

#Export para revisar
df_merge1.to_csv('./export/pruebas/ej_merge1.csv')
df_merge2.to_csv('./export/pruebas/ej_merge2.csv') 
df_texto1.to_csv('./export/pruebas/ej_texto1.csv')
df_texto2.to_csv('./export/pruebas/ej_texto2.csv')


# EXTRACCION DE CARACTERISTICAS

# LabelEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#df_inicial.to_excel("antes.xlsx")

# Normalizar Topic y Source
df_inicial.Topic = le.fit_transform(df_inicial.Topic)
df_inicial.Source = le.fit_transform(df_inicial.Source)

#df_inicial.to_excel("despues.xlsx")
    
# TF-IDF ¿?


# LightGBM




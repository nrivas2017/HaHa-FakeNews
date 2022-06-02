import matplotlib.pyplot as plt
import numpy as np
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords 
import math
import pandas as pd 
import seaborn as sns
from pylab import savefig

#Analisis exploratorio de datos (EDA)
def EDA(df):

    #Comprobacion de datos nulos y tipos de variables por columna
    infoData = df.info()
    print(infoData)

    #Count categoria
    fig1 = sns.countplot(data=df, x='Category')
    plt.title("Frecuencia de noticias por Category")
    plt.show()
    fig1 = fig1.get_figure()  
    fig1.savefig('./export/graficos/g1.png')

    #Count topic
    fig2 = sns.catplot(y='Topic',
                kind='count',
                height=6, 
                aspect=1,
                order=df.Topic.value_counts().index,
                data=df)
    plt.title("Frecuencia de noticias por Topic")
    plt.show()
    fig2.savefig('./export/graficos/g2.png')

    #Fake/True por categoria
    fig3 = sns.catplot(y="Topic", hue="Category", kind="count", edgecolor=".6",
                data=df)
    plt.title("Veracidad de noticias por Category")
    plt.show()
    fig3.savefig('./export/graficos/g3.png')

    # #Fake/True por fuente
    # sns.catplot(y="Source", hue="Category", kind="count", edgecolor=".6",
    #             data=df)
    # plt.title("Veracidad de noticias por Fuente")
    # plt.show()


    return 




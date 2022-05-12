import matplotlib.pyplot as plt
import numpy as np
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords 
import math
import pandas as pd 


def crear_mFrecuencia(sentences):
    mFrecuencia = {}
    stopWords = set(stopwords.words("spanish"))
    ps = PorterStemmer() # Libreria de ntlk

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        mFrecuencia[sent[:15]] = freq_table

    return mFrecuencia

def crear_mTermFrequency(mFrecuencia):
    mTF = {}
    for sent,f_table in mFrecuencia.items():
        tf_table = {}

        cantidad_palabras = len(f_table)
        for token,count in f_table.items():
            tf_table[token] = count / cantidad_palabras

        mTF[sent] = tf_table
    return mTF

def crear_OracionexPal(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table
def crear_mIDF(freq_matrix,OraxPal,total_doc):
    mIDF = {}
    for sent,f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_doc / float(OraxPal[word]))

        mIDF[sent] = idf_table
    return mIDF
def crear_mTFIDF(tf_matrix,idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix
def puntuar_palabra(tf_idf_matrix) -> dict:

    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue
def plotData(df):
    
    #nlp = spacy.load("es_core_news_lg")
    texto = df['Text'][0]

    #1. Tokenizar el texto
    sentences = sent_tokenize(texto)
    largo_documento = len(sentences)


    #2. Crear la matrix de frecuencia
    matrix_freq = crear_mFrecuencia(sentences)
    #print(matrix_freq)

    #3. Calcular el "Term Frecuency" generado de la matrix

    # TF(t) = Numero de veces que aparece un termino en (t) / Numero total de terminos en el documento 

    mTF = crear_mTermFrequency(matrix_freq)
    #print(mTF)
    #4. Calcular cuantas oraciones contiene una palabra
    OracionesxPal = crear_OracionexPal(mTF)
    #data_plot = []

    #5. calcular matriz IDF (Inverse document frequency)
     # IDF(t) =log_e(Numero de veces que aparece un termino en (t) / Numero total de terminos en el documento)
    mIDF = crear_mIDF(matrix_freq,OracionesxPal,largo_documento)
    #print(mIDF)

    #6. Calcular matriz TF-IDF

    TFIDF= crear_mTFIDF(mTF,mIDF)

    #7. Puntuar palabras

    score = puntuar_palabra(TFIDF)
    #print(score)

    puntuaciones = []
    for item in score :
        puntuaciones.append(score[item])

    max_y= (len(puntuaciones))*0.1

    #Calcular Y 
    y = []
    y_sum = 0
    for i in range(len(puntuaciones)):
        y.append(y_sum)
        y_sum +=0.1
    plt.scatter(puntuaciones,y)
    plt.show()


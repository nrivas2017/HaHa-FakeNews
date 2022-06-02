import spacy 
import pandas as pd
import json 
import numpy as np
import matplotlib.pyplot as plt 
import textacy
from textacy import representations
from textacy import extract
from os.path import exists
from functools import partial


def exceltoJson(name):
	df = pd.read_excel("corpus/"+name)
	dic = []
	keys = []
	#Se guarda los nombre de columnas
	for i in df.columns:
		keys.append(i)
	for i in range(len(df)):
		jsonItem ={}
		for key in keys:
			if(key == 'Id'):
				np_int = np.int64(df[key][i])
				jsonItem[key.lower()] = np_int.item()			
			else:				
				jsonItem[key.lower()] = df[key][i]
		dic.append(jsonItem)
	name = name.split(".")[0] + ".json"
	if(exists(name) == False):
		with open(name,"w") as f:
			json.dump(dic,f,indent=4)
		print("Archivo guardado !! ")
	return dic
def Ner(doc):
	a = []
	for ent in doc.ents:
		a.append(ent)
	return a 
def SpacyExample(data):
	nlp = spacy.load("es_core_news_lg")
	noticia = data[1]['text']

	print("noticia completa : "+"\n" + noticia)
	print("-------------------------------------------")
	doc = nlp(noticia)

	ents = doc.ents
	aEnts = []
	for ent in ents:
		i = ent.start
		f = ent.end
		aEnts.append([ent,(i,f)])

	aTokens = []
	save = True


	#Suma el postag con ner 
	for i in range(len(doc)) :
		for ent in aEnts:
			entidad = ent[0]
			eI = ent[1][0]
			eF = ent[1][1]
			if(i == eI):
				dif = eF - eI
				if(dif == 1):
					aTokens.append(entidad)
					save=False
				else:
					aTokens.append(entidad)
					i = eF+1
					save = False
		if(save):
			aTokens.append(doc[i])
		save = True

####



# POSTAG + NER JUNTOS 
#for token in aTokens:
#	vNorm.append(token.vector_norm)
	#if(type(token) != spacy.tokens.span.Span):	
	#	print(token.tag_)
	#else:
		#print(token.label_)

#Obtener n-grams
#Obtiene las palabras que mas se repiten 


#Se cargan las noticias 
data = exceltoJson("development.xlsx")

#Postag+Ner con spacy
#SpacyExample()

def textacyExample(data):

	texto = data[1]['text']


	#Se carga la noticia
	doc = textacy.make_spacy_doc(texto,"es_core_news_lg")


	docs_terms =  (extract.terms( d,
		ngs=partial(extract.ngrams, n=2, include_pos={"NOUN","ADJ"}), 
		ents=partial(extract.entities, include_types={"PERSON","ORG","GPE","LOC"})) for d in doc )
	#TermFrequency
	#doc_term_matrix, vocab = representations.build_doc_term_matrix(doc, tf_type="linear", idf_type="smooth")

	tokenized_docs = (extract.terms_to_strings(doc_terms, by="lemma") for doc_terms in docs_terms)

	doc_term_matrix, vocab = representations.build_doc_term_matrix(tokenized_docs)

	print(doc_terms_matrix)


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
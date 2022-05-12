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

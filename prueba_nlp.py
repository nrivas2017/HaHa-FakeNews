import spacy 
import pandas as pd
import json 
import numpy as np
from os.path import exists


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
def PostTag(doc):
	aPost = []	
	for token in doc : 
		aPost.append(token)
	return aPost

def Ner(doc):
	a = []
	for ent in doc.ents:
		a.append(ent)
	return a 
data = exceltoJson("development.xlsx")
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
for token in aTokens:
	if(type(token) != spacy.tokens.span.Span):
		
		print(token.tag_)
	else:
		print(token.label_)
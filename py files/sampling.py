# -*- coding: utf-8 -*-
"""IR_Tag_Detector.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G05WZnbii4XiYVxXKUAYYGLD7QFxlsq3
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pickle
import os
import chardet
import pandas as pd
import numpy as np
import nltk
import string
import copy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from bs4 import BeautifulSoup as bs
from scipy.spatial.distance import cosine

import re
import keyword
import heapq
from nltk.tokenize import RegexpTokenizer
import math
import codecs

path = 'gdrive/My Drive/Datasets/StackOverflow/'

os.chdir(path)

#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z

def find_tags(tags,number):
    tags.sort()
    tags.reverse()
#     print(tags)
    sum1=0
    for i in range(0,number):
#         print(tags[i])
        sum1+=tags[i]
    avg=sum1/float(number)
#     print(avg)
    i=0
    counter=0
    while avg<tags[i]:
        counter+=1
        i+=1
    return counter

def magnitude(array_list):
    s=0
    for i in range(len(array_list)):
        s=s+(array_list[i]**2)
    return np.power(s,0.5)

def cosine_vec(query_vector,vec2):
    query_vector=np.array(query_vector)
    vec2=np.array(vec2)
    q1=magnitude(list(query_vector))
    cosine_similarity_val=np.dot(query_vector,vec2)/float(q1*magnitude(list(vec2)))

def cosine(a, b):
    m = a.multiply(b)
    numerator = m.sum(axis=1)
    det_a = np.sqrt(a.sum(axis = 1))
    det_b = np.sqrt(b.sum(axis = 1))
    return numerator / (det_a * det_b)

def preprocess(pd):
    pd = pd.str.lower()
    pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])
    pd = pd.apply(lambda x: [item for item in x if item not in removing_words])
    pd = pd.apply(lambda x: [stemmer.stem(y) for y in x])
    pd = pd.str.join(' ')
    pd = pd.str.replace('[{}]'.format('$<>?@`\'"'), ' ')
    return pd

def extract_code(m):
    b = bs(m)
    t = b.find_all('code')
    for tag in t:
        tag.replace_with('')
    k = []
    for i in t:
        z = str(i).replace("<code>", "")
        z = str(z).replace("</code>", "")
        k.append(z)
    return k

def extract_body(m):
    b = bs(m)
    t = b.find_all('code')
    for tag in t:
        tag.replace_with('')
    a = list({tag.name for tag in b.find_all()})
    for i in a:
        b = str(b).replace("<"+str(i)+">", " ")
        b = str(b).replace("</"+str(i)+">", " ")
    return b

def determineWordType(t1, t2):
   keylist = keyword.kwlist
   keys1 = 0
   keys2 = 0
#     func1 = 0
#     func2 = 0
   cf1 = 0
   cf2 = 0
   vars1 = 0
   vars2 = 0
   i = 0
   while i < len(t1):
       if t1[i] in keylist:
           keys1 += 1
#         elif re.search(r'[a-zA-Z][a-zA-z0-9]*\(*\):$', t1[i]):
#             func1 += 1
       elif re.search(r':$', t1[i]):
           cf1 += 1
       else:
           vars1 += 1
       i += 1
   i = 0
   while i < len(t2):
       if t2[i] in keylist:
           keys2 += 1
#         elif re.search(r'[a-zA-Z][a-zA-z0-9]*\(*\):$', t2[i]):
#             func2 += 1
       elif re.search(r':$', t2[i]):
           cf2 += 1
       else:
           vars2 += 1
       i += 1
   return [keys1, cf1, vars1], [keys2, cf2, vars2]

def matchCode(a, b):
   simscore = 0
   l1 = a.splitlines()
   l2 = b.splitlines()
#     print(l1)
#     print(l2)
   i = 0
   m = len(l1)
   n = len(l2)
   while i < min(m,n):
       t1=l1[i].split(" ")
       t2=l2[i].split(" ")
       vec1, vec2 = determineWordType(t1, t2)
       difvec = np.subtract(vec1, vec2)
       if sum(abs(difvec)) < min(len(t1), len(t2)):
           simscore += 1
       i += 1
   if max(m,n)==0:
     return 0.0
   return simscore/max(m,n)

import nltk
nltk.download('stopwords')

stop_words = stopwords.words('english')
needed_words = ['what', 'which', 'if', 'while', 'for', 'between', 'into', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over','then','not','how','do']
removing_words = list(set(stop_words).difference(set(needed_words)))

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

# df = pd.read_csv("Questions.csv", encoding = "ISO-8859-1")
# print(df.head())

# df1 = pd.read_csv("Tags.csv", encoding = "ISO-8859-1")
# print(df1.head())

# df2 = pd.read_csv("Answers.csv", encoding = "ISO-8859-1")
# print(df2.head())

# processed = copy.deepcopy(df)

ans_processed = copy.deepcopy(df2)

import nltk
nltk.download('wordnet')

# processed['Title'] = preprocess(processed['Title'])
# processed['Code'] = processed['Body'].apply(lambda x: extract_code(x))
# processed['Body'] = preprocess(processed['Body'].apply(lambda x: extract_body(x)))
# processed['Id']=df['Id']
# processed.head()

ans_processed['Code'] = ans_processed['Body'].apply(lambda x: extract_code(x))
ans_processed['Body'] = preprocess(ans_processed['Body'].apply(lambda x: extract_body(x)))
ans_processed['Id']=df2['Id']
ans_processed.head()

ans_processed

del df2

# pickle.dump(ans_processed, open('ans_processed.pickled', 'wb'))

# with open('ans_processed.pickled', 'rb') as reader:
#     ans_processed = pickle.load(reader)

# #Convert everything into lists
# id3=processed['Id']
# body=processed['Body']
# title=processed['Title']
# code=processed['Code']
# id3=list(id3)
# body=list(body)
# title=list(title)
# code=list(code)

# len(code)

# code[8][0].split('\n')

# processed['Code'][100]

# len(processed['Code'])

"""### Load processed pickle file"""

import pickle

# pickle.dump(processed, open('processed.pickled', 'wb'))

with open('processed.pickled', 'rb') as reader:
    processed = pickle.load(reader)

processed.head()

"""### Filter python code"""

def locatePCode(code):
    keywords = ['for', 'while', 'if', 'elif', 'else', 'try', 'except', 'def', 'from', 'class']
    l = code.splitlines()
#     print(l)
    n = len(l)
    c = 0
    for i in l:
        w = i.split(" ")
#         print(w)
        for j in w:
            if j != ' ':
                if j in keywords:
                    c += 1
                break
#         print(c)
    if c > 0:
        return True
    else:
        return False

# x = ['class MyStruct():\n    def __init__(self, field1, field2, field3):\n        self.field1 = field1\n        self.field2 = field2\n        self.field3 = field3\n']
# locatePCode(''.join(x))
# processed['Id']

def filterCode(code, Id, df):
    selec_code = []
    n = 0
    for i in code:
        if i == []:
            selec_code.append([Id[n], df['Score'][n], df['Title'][n], df['Body'][n], i])
        else:
            flag = locatePCode(''.join(i))
            if flag is True:
                selec_code.append([Id[n], df['Score'][n], df['Title'][n], df['Body'][n], i])
        n += 1
    return selec_code

selec_code = filterCode(processed['Code'], processed['Id'], processed)

# pickle.dump(selec_code, open('pfiltered_ques.pickled','wb'))

with open('pfiltered_ques.pickled', 'rb') as reader:
    selec_code = pickle.load(reader)

len(selec_code)

selec_code = np.array(selec_code)

selec_code.shape

def filterAnswers(Id, ParId, df):
    selec_ans = []
    n = 0
    for i in ParId:
        if i in Id:
            selec_ans.append([df['Id'][n], df['ParentId'][n], df['Score'][n], df['Body'][n], df['Code'][n]])
        n += 1
    return selec_ans

selec_ans = filterAnswers(set(selec_code[:,0]), ans_processed['ParentId'], ans_processed)

selec_ans[0][4]

pickle.dump(selec_ans, open("pfiltered_ans.pickled", 'wb'))

"""### Processing Tags, etc."""

qId = selec_code[:,0]

len(qId)

qId[:10]

df1.head(20)

# selec_tags = []
# n = 0
# for i in qId:
#     flag = False
#     while i == df1['Id'][n]:
#         selec_tags.append([df1['Id'][n], df1['Tag'][n]])
#         n += 1
#         flag = True
#     if flag is False:
#         n += 1

selec_tags = []
qId = set(qId)
for i in range(len(df1['Id'])):
    if df1['Id'][i] in qId:
        selec_tags.append([df1['Id'][i], df1['Tag'][i]])

selec_tags[0]

pickle.dump(selec_tags, open('pfiltered_tags.pickled','wb'))

def filterTags(Id, Tags):
    ftags = []
    n = 0
    for i in Tags:
        if i == 'python':
            ftags.append(Id[n])
        n += 1
    return ftags

ftags = filterTags(df1['Id'], df1['Tag'])

len(ftags)

questions_with_hash={}
for i in range(len(id3)):
    temp={}
    temp['body']=body[i]
    temp['title']=title[i]
    temp['code']=code[i]
    questions_with_hash[id3[i]]=temp



id1=df1['Id']
id1=list(id1)
tag=df1['Tag']
tag=list(tag)

# tags_unique=list(set(tag))



quesid_invertedindex={}
for i in range(len(id1)):
    if tag[i] not in quesid_invertedindex.keys():
        temp=[]
        temp.append(id1[i])
        quesid_invertedindex[tag[i]]=temp
    else:
        quesid_invertedindex[tag[i]].append(id1[i])

flag={}
for i in quesid_invertedindex.keys():
    flag[i]=0



tag_list={}
counter=0
for i in range(len(tag)):
    counter+=1
    if tag[i]=='python' or tag[i]=='python-2.7' or tag[i]=='python-3.0':
        continue
    if counter%50000==0:
        print("Counter : ",counter)
    try:
        flag2=tag_list[tag[i]]
        temp_t=questions_with_hash[id1[i]]     
        tag_list[tag[i]]['body']+=temp_t['body']
        tag_list[tag[i]]['title']+=temp_t['title']
        tag_list[tag[i]]['code'].append(temp_t['code'])
        flag[tag[i]]+=1
                         
    except:
        temp_t=questions_with_hash[id1[i]]
        temp={}
        temp['body']=temp_t['body']
        temp['title']=temp_t['title']
        temp1=[]
        temp1.append(temp_t['code'])
        temp['code']=temp1
        tag_list[tag[i]]=temp
        flag[tag[i]]+=1



print("...........................User Query..........................")
query_body=input("Enter the Question Body : ")
query_code=input("Enter the Question Code")

#Assigning weights to zonal section
weight_title=6
weight_body=4
weight_code=3

tokenizer=RegexpTokenizer('\s+',gaps=True)



body_corpus = pickle.load(open("body.txt", "rb") )
title_corpus = pickle.load(open("title.txt", "rb" ) )



#It needed when body.txt and title.txt deleted. 

# title_corpus=''
# body_corpus=''
# counter=0
# for i in range(len(title)):
#     counter+=1
#     if counter%20000==0:
#         print("Counter : ",counter)
#     temp=tokenizer.tokenize(title[i])
#     temp1=tokenizer.tokenize(body[i])
#     for j in range(len(temp)):
#         title_corpus+=" "+ temp[j]
#     for j in range(len(temp1)):
#         body_corpus+=" "+ temp1[j]

with open("tag_list.txt", 'wb') as f:
    pickle.dump(tag_list, f)

with open("body.txt", 'wb') as f:
    pickle.dump(body_corpus, f)

with open("title.txt", 'wb') as f:
    pickle.dump(title_corpus, f)



title_corpus_main=[]
title_corpus_main.append(title_corpus)

body_corpus_main=[]
body_corpus_main.append(body_corpus)

#For Title using question body
vc_title_tag = CountVectorizer() #TfidfVectorizer()
vec_title = vc_title_tag.fit_transform(title_corpus_main)
q_title = vc_title_tag.transform(preprocess(pd.DataFrame([query_body])[0]))



#For body using question bod
vc_body_tag = CountVectorizer() #TfidfVectorizer()
vec_body = vc_body_tag.fit_transform(body_corpus_main)
q_title1 = vc_body_tag.transform(preprocess(pd.DataFrame([query_body])[0]))







#For body title of tagslist in which I have the body, tiltle and code for each tags at one place.
tokenizer = RegexpTokenizer(r'\w+')
similarity_body=[]
similarity_title=[]
similarity_code=[]
tag_similar=[]
counter=0
for i in tag_list.keys():
    counter+=1
    if counter%2000==0:
        print("Counter : ",counter)
    a=[]
    a.append(tag_list[i]['body'])
    sim1=vc_body_tag.transform(a)
    similarity_body.append(pairwise_distances(sim1, q_title1)[0][0])
    a=[]
    a.append(tag_list[i]['title'])
    sim1=vc_title_tag.transform(a)
    similarity_title.append(pairwise_distances(sim1, q_title)[0][0])    
    tag_similar.append(i)
#     sim1=matchCode(tag_list[i]['code'],query_code)



rel=int(input("Enter the number of relevant tags : "))
main_score=[]
for i in range(len(similarity_body)):
    main_score.append(similarity_body[i]*weight_body+similarity_title[i]*weight_title)
a=copy.deepcopy(np.array(main_score))



rel_opt=find_tags(list(a),rel)



#Another approach to take top relevant tags
list_top=heapq.nlargest(rel_opt, range(len(a)), a.take)

print("More Similar tags for give Question are : ",list_top)

tags1=[]
for i in range(len(list_top)):
    tags1.append(tag_similar[list_top[i]])
    print(tag_similar[list_top[i]])





#List of Question IDs most relevant to user questions
ques_id=[]
for i in range(len(tags1)):
    ques_id+=quesid_invertedindex[tags1[i]]

print("Questions ID are : ")
print(ques_id)

quesid_invertedindex['numpy']



"""## Assisting code matching"""

import re

def replaceVariables(code):
    n_code = []
    keylist = keyword.kwlist
    lines = code.splitlines()
    for l in lines:
        arr = l.split(" ")
        for (i, word) in enumerate(arr):
            if re.match(r'[a-zA-Z][a-zA-Z0-9]*', word):
                if word not in keylist:
                    arr[i] = 'var'
            else:
                arr[i] = word
        n_code.append(''.join(arr))
    return ''.join(n_code)
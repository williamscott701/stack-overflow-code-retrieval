
# coding: utf-8

# In[32]:


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


# In[33]:


import re
import keyword
import heapq
from nltk.tokenize import RegexpTokenizer
import math
import codecs
from networkx.algorithms.similarity import graph_edit_distance
import networkx as nx
from numpy import linalg as LA


# In[34]:


#https://stackoverflow.com/questions/12122021/python-implementation-of-a-graph-similarity-grading-algorithm
def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)


# In[35]:


def adjacency_create(edges):
    unique=[]
    for (i,j) in edges:
        if i not in unique:
            unique.append(i)
        if j not in unique:
            unique.append(j)
    vertex1={}
    for j in range(len(unique)):
        vertex2={}
        for i in range(len(unique)):
            
            vertex2[unique[i]]=0
        vertex1[unique[j]]=vertex2
    for (i,j) in edges:
        vertex1[i][j]+=1
    adj=[]
    for i in vertex1.keys():
        temp=[]
        for j in vertex1[i].keys():
            temp.append(vertex1[i][j])
        adj.append(temp)
    return np.array(adj)
    


# In[36]:


def steady_sim_neigh(A,B):
    G1=nx.from_numpy_matrix(A)
    G2=nx.from_numpy_matrix(B)
    
    w1, v1 = LA.eig(np.matmul(np.transpose(A),np.array(A)))
    v1=v1.transpose()
    sim1=np.array(np.matmul(np.matrix(A),np.matrix(v1)))
    w1, v1 = LA.eig(np.matmul(np.transpose(B),np.array(B)))
    v1=v1.transpose()
    sim2=np.array(np.matmul(np.matrix(B),np.matrix(v1)))
    a1=LA.norm(sim1)
    a2=LA.norm(sim2)
    a=a1+a2
    
    w1, v1 = LA.eig(np.matmul(np.array(A),np.transpose(A)))
    v1=v1.transpose()
    sim1=np.array(np.matmul(np.matrix(A),np.matrix(v1)))
    w1, v1 = LA.eig(np.matmul(np.array(B),np.transpose(B)))
    v1=v1.transpose()
    sim2=np.array(np.matmul(np.matrix(B),np.matrix(v1)))
    b1=LA.norm(sim1)
    b2=LA.norm(sim2)
    b=b1+b2
    return (0.5*np.absolute(a1-a2)/float(a)+0.5*np.absolute(b1-b2)/float(b))


# In[37]:


def steady_sim(A,B):
    G1=nx.from_numpy_matrix(A)
    G2=nx.from_numpy_matrix(B)
    w1, v1 = LA.eig(np.array(A))
    v1=v1.transpose()
    sim1=np.array(np.matmul(np.matrix(A),np.matrix(v1)))
    w1, v1 = LA.eig(np.array(B))
    v1=v1.transpose()
    sim2=np.array(np.matmul(np.matrix(B),np.matrix(v1)))
    a1=LA.norm(sim1)
    a2=LA.norm(sim2)
    a=a1+a2
    return np.absolute(a1-a2)/float(a)


# In[38]:


def scaling(lap):
    sum=np.sum(lap)
    lap1=[lap[i]/float(sum) for i in range(len(lap))]
    return lap


# In[39]:


def graph_similarity_measure(a,b):
    A=np.matrix(a)
    B=np.matrix(b)
    G1=nx.from_numpy_matrix(A)
    G2=nx.from_numpy_matrix(B)
    #Similarity 1
    if a.shape==b.shape:
        edit_d=nx.graph_edit_distance(G1, G2)
    else:
        edit_d=0
    #Similarity2
    iso=nx.is_isomorphic(G1,G2)
    d1=np.sum(np.array(A))
    d2=np.sum(np.array(B))

    
    d3=max([d1,d2])
    if iso:
        return 1
        
    laplacian1 = nx.spectrum.laplacian_spectrum(G1)
    laplacian2 = nx.spectrum.laplacian_spectrum(G2)
    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)
    #Similarity 3
    lap1=scaling(laplacian1[:k])
    lap2=scaling(laplacian2[:k])
    eig_similarity = sum((lap1 - lap2)**2)/float(k)
    
    #Similarity 4
    steady_similarity=steady_sim(A,B)
    #Similarity 5
    steady_similarity1=steady_sim_neigh(A,B)
    return (0.25*(1-steady_similarity)+0.25*(1-eig_similarity) +0.25*(edit_d/float(d3))+0.25*(1-steady_similarity1))


# In[40]:


def matchCode(a, b):
    simscore = 0
    l1 = a.splitlines()
    l2 = b.splitlines()
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


# In[41]:


def matchSlidingWindow(a, b):
    simscores = []
    l1 = a.splitlines()
    l2 = b.splitlines()
    m = len(l1)
    n = len(l2)
    wsize = min(m, n)
    i = 0
    
    if m < n:
        i += m
        k = 0
        while i <= n:
            simscores.append(matchCode(''.join(l1), ''.join(l2[k:i])))
            k += 1
            i += 1      
    else:
        i += n
        k = 0
        while i <=m:
            simscores.append(matchCode(''.join(l1[k:i]), ''.join(l2)))
            k += 1
            i += 1
    return max(simscores)


# In[42]:


def determineWordType(t1, t2):
    keylist = keyword.kwlist
    keys1 = 0
    keys2 = 0
    cf1 = 0
    cf2 = 0
    vars1 = 0
    vars2 = 0
    i = 0
    while i < len(t1):
        if t1[i] in keylist:
            keys1 += 1
        elif re.search(r':$', t1[i]):
            cf1 += 1
        else:
            vars1 += 1
        i += 1
    i = 0
    while i < len(t2):
        if t2[i] in keylist:
            keys2 += 1
#         elif re.search(r'[a-zA-Z][a-zA-z0-9]\(\):$', t2[i]):
#             func2 += 1
        elif re.search(r':$', t2[i]):
            cf2 += 1
        else:
            vars2 += 1
        i += 1
    return [keys1, cf1, vars1], [keys2, cf2, vars2]


# In[43]:


#geeks for geeks 
def sort_list(list1, list2): 
  
    zipped_pairs = zip(list2, list1) 
  
    z = [x for _, x in sorted(zipped_pairs)] 
      
    return z 


# In[44]:


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


# In[45]:


def magnitude(array_list):
    s=0
    for i in range(len(array_list)):
        s=s+(array_list[i]**2)
    return np.power(s,0.5)


# In[46]:


def cosine_vec(query_vector,vec2):
    query_vector=np.array(query_vector)
    vec2=np.array(vec2)
    q1=magnitude(list(query_vector))
    cosine_similarity_val=np.dot(query_vector,vec2)/float(q1*magnitude(list(vec2)))


# In[47]:


def cosine(a, b):
  m = a.multiply(b)
  numerator = m.sum(axis=1)
  det_a = np.sqrt(a.sum(axis = 1))
  det_b = np.sqrt(b.sum(axis = 1))
  return numerator / (det_a * det_b)


# In[48]:


def preprocess(pd):
  pd = pd.str.lower()
  pd = pd.apply(lambda x: [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(x)])
  pd = pd.apply(lambda x: [item for item in x if item not in removing_words])
  pd = pd.apply(lambda x: [stemmer.stem(y) for y in x])
  pd = pd.str.join(' ')
  pd = pd.str.replace('[{}]'.format('$<>?@`\'"'), ' ')
  return pd


# In[49]:


def dash_improve(str1):
    s=[]
    s1=''
    for i in range(len(str1)):
        s.append(str1[i])
#     print(s)

    for i in range(len(s)):
        if s[i]!='-':
            s1=s1+s[i]
    return s1


# In[50]:


def digit_improve(str1):
    s=[]
    s1=''
    for i in range(len(str1)):
        s.append(str1[i])
#     print(s)
    for i in range(len(s)):
        if s[i]=='0':
            s1=s1+'zero'
        if s[i]=='1':
            s1=s1+'one'
        if s[i]=='2':
            s1=s1+'two'
        if s[i]=='3':
            s1=s1+'three'
        if s[i]=='4':
            s1=s1+'four'
        if s[i]=='5':
            s1=s1+'five'
        if s[i]=='6':
            s1=s1+'six'
        if s[i]=='7':
            s1=s1+'seven'
        if s[i]=='8':
            s1=s1+'eight'
        if s[i]=='9':
            s1=s1+'nine'
        if s[i]>='0' and s[i]<='9':
            pass
        else:
            s1=s1+s[i]
    return s1


# In[51]:


def punc_improve(str1):
    s=[]
    s1=''
    for i in range(len(str1)):
        s.append(str1[i])
#     print(s)

    for i in range(len(s)):
        if (s[i]=='.' or s[i]==',' or  s[i]=='!'or  s[i]=='*' or s[i]=='+'  or 
            s[i]=='-' or s[i]=='\"'  or s[i]=='\'' or
            s[i]=='{' or s[i]=='}' or s[i]==';' or s[i]==':' or s[i]=='(' or
            s[i]==')' or s[i]=='='  or s[i]=='@' or s[i]=='>' or s[i]=='[' or 
            s[i]==']' or s[i]=='|' or s[i]=='#' or s[i]=='%' or s[i]=='`' or 
            s[i]=='~' or s[i]=="/" or s[i]=='_' or s[i]=='<' or s[i]=='?' or  
            s[i]==' ' or s[i]=='$' or s[i]=='^'or s[i]=='' or s[i]==' ' or s[i]=='&'):
            pass
        else:
            s1=s1+s[i]
        
    return s1


# In[52]:


def preprocess_for_query(pd):
    temp=[]
    temp=tokenizer.tokenize(pd)
    temp =[w.lower() for w in temp]
    for n in range(len(temp)):
        t1=str(dash_improve(str(temp[n])))

        temp[n]=t1
        t2=str(digit_improve(str(temp[n])))

        temp[n]=t2
        t3=str(punc_improve(str(temp[n])))

        temp[n]=t3
    return temp


# In[53]:


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


# In[54]:


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


# In[55]:


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


# In[56]:


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


# In[57]:


def replaceVariables(code):
    n_code = []
    keylist = keyword.kwlist
    lines = code.splitlines()
    
    for l in lines:
        flag=False
        arr = l.split(" ")
        
        for (i, word) in enumerate(arr):
            if re.match(r'\s*#', word):
                flag=True
                break
            if re.match(r'[a-zA-Z][a-zA-Z0-9]*', word):
                if word not in keylist:
                    arr[i] = 'var'
            else:
                arr[i] = word
        if flag:
            n_code.append(' ')
        else:
            n_code.append(' '.join(arr))
    return '\n'.join(n_code)


# In[58]:


import copy

def print_code(a):
    for idx, i in enumerate(a):
        print(i, idx)
    print("")

def get_ws(line):
    c = 0
    for i in line:
        if i in ' ':
            c+=1
        else:
            return c



def find_edges_for(states, end):
    edges = []
    
    edges.append((hash(str(states[0])), hash(str(states[1][0]))))
    edges.append((hash(str(states[1][-1])), hash(str(states[0]))))
    edges.append((hash(str(states[0])), hash(str(end))))

    return edges

def get_for(lines):
#     print_code(lines)

    main_stack = []
    stack = []
    
    ws = get_ws(lines[0])

    index = 0
    edges = []
    
    main_stack.append(lines[index])

    index += 1

    while True:
        if len(lines) == index:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
            return main_stack,  find_edges_for(main_stack, end) + edges, index
        
        try:
            ws_inside = get_ws(lines[index])
        except:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
            return main_stack, find_edges_for(main_stack, end) + edges, index
        
        if ws_inside == ws:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
#             if get_ws(lines[index]) == ws_inside:
            try:
                end = lines[index]
            except:
                pass
            return main_stack, find_edges_for(main_stack, end) + edges, index

        if ws_inside is None:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
            return main_stack,  find_edges_for(main_stack, end) + edges, index
        
        if ws_inside > ws:
            states, edges_, length = get_general(lines[index:])
            edges += edges_
            index += length
            stack += states
        
        index += 1

    return main_stack, [], 0

def get_if(lines):
#     print_code(lines)
    lines.append("\n")

    main_stack = []
    
    ws = get_ws(lines[0])

    if_heads = []
    if_s = []
    
    else_reached = False
    
    p = []
    
    c = 0
    for i in lines:
        ws_i = get_ws(i)
        
        if ws_i is None:
            break
        
        if ws_i == ws:
            if not(i[ws_i:].startswith("if") or i[ws_i:].startswith("elif") or i[ws_i:].startswith("else")):
                if_s.append(p)
                break
            
            if_heads.append(i)
            if len(p) > 0:
                if_s.append(p)
                p = []
        
        if ws_i > ws:
            p.append(i)
        
        c+=1
    if len(p) > 0:
        if_s.append(p)
    
    states = []
    edges = []
    for i, j in zip(if_heads, if_s):
        states.append(i)
        states_, edges_, _ = get_general(j)
        states.append(states_)
        edges += edges_
    
    for idx, i in enumerate(states):
        if idx%2 == 0:
            try:
                edges.append((hash(str(states[idx])), hash(str(states[idx+1]))))
            except:
                pass
            try:
                edges.append((hash(str(states[idx])), hash(str(states[idx+2]))))
            except:
                pass
        else:
            try:
                edges.append((hash(str(states[idx])), hash(str(end))))
            except:
                pass
    return states, edges, c-1

# Processing General

def find_edges_general(states, end):
    edges = []
    
    edges.append((hash(str(states)), hash(str(states[0]))))
    
    for idx, i in enumerate(states):
        try:
            edges.append((hash(str(states[idx])), hash(str(states[idx+1]))))
        except:
            pass
    try:
        edges.append((hash(str(states[-1])), hash(str(end))))
    except:
        pass
    return edges

def get_general(lines):
    main_stack = []
    stack = []

    ws = get_ws(lines[0])

    index = 0
    edges = []

    while True:
        if len(lines) == index:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
            return main_stack, find_edges_general(main_stack, end) + edges, index
        
        ws_inside = get_ws(lines[index])

        if ws == ws_inside:
            try:
                if(lines[index][ws : ws+2] == "if" and get_ws(lines[index])>ws):
                    states, edges_, length = get_if(lines[index:])
                    index += length - 1
                    edges += edges_
                    main_stack.append(states)
                elif (lines[index][ws:].startswith("for") or lines[index][ws:].startswith("while")) and get_ws(lines[index])>ws:
                    if index == len(lines)-1:
                        main_stack.append(stack)
                        main_stack.append(lines[index])
                        end = ""
                        return main_stack, find_edges_general(main_stack, end) + edges, index
                    states, edges_, length = get_for(lines[index:])
                    index += length - 1
                    edges += edges_
                    main_stack.append(states)
                    if index == len(lines):
                        end = ""
                        return main_stack, find_edges_general(main_stack, end) + edges, index
                else:
                    main_stack.append(lines[index])
            except:
                main_stack.append(lines[index])
        if ws_inside is None:
            if len(stack) > 0:
                main_stack.append(stack)
            end = ""
            return main_stack, find_edges_general(main_stack, end) + edges, index
        
        if ws_inside > ws:
            states, edges_, length = get_general(lines[index:])
            index += length
            edges += edges_
            main_stack.append(states)
            if len(lines) == index:
                if len(stack) > 0:
                    main_stack.append(stack)
                end = ""
                return main_stack, find_edges_general(main_stack, end) + edges, index

        if ws_inside < ws:
            end = ""
            if len(stack) > 0:
                main_stack.append(stack)
                stack = []
                if get_ws(lines[index]) == ws:
                    end = lines[index]
            return main_stack, find_edges_general(main_stack, end) + edges, index-1

        index += 1
    
def get_all_edges(c):
    states, edges, length = get_general(c)
    return edges


# In[59]:


# e=get_all_edges(code[2][0].split('\n'))
# d=adjacency_create(e)
# f=get_all_edges(code[3][0].split('\n'))
# h=adjacency_create(f)
# graph_similarity_measure(d,d)


# In[60]:


stop_words = stopwords.words('english')
needed_words = ['what', 'which', 'if', 'while', 'for', 'between', 'into', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over','then','not','how','do']
removing_words = list(set(stop_words).difference(set(needed_words)))


# In[61]:


lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()


# In[62]:


import pickle
tag_detail = pickle.load(open("pfiltered_tags.pickled","rb"))
questions_only= pickle.load(open("pfiltered_ques.pickled","rb"))


# In[63]:


df= pd.DataFrame(questions_only)
print(df.head())


# In[64]:


df1= pd.DataFrame(tag_detail)
print(df1.head())


# In[65]:


# df1[df1['Tag'].isin([ 'pandas'])].drop_duplicates(subset=['Id'], keep='last').shape


# In[66]:


processed = copy.deepcopy(df)


# In[67]:


df.head()


# In[68]:


#Convert everything into lists
id3=processed[0]
body=processed[3]
title=processed[2]
code=processed[4]
id3=list(id3)
body=list(body)
title=list(title)
code=list(code)


# In[69]:


questions_with_hash={}
for i in range(len(id3)):
    temp={}
    temp['body']=body[i]
    temp['title']=title[i]
    temp['code']=code[i]
    questions_with_hash[id3[i]]=temp


# In[70]:


id1=df1[0]
id1=list(id1)
tag=df1[1]
tag=list(tag)


# In[71]:


unique_list=list(set(tag))


# In[72]:


# tag_check=[]
# counter=0
# for i in range(len(tag)):
#     counter+=1
#     try:
#         tag_check.append(preprocess(pd.DataFrame([tag[i]])[0])[0])
#     except:
#         pass
    


# In[73]:


# with open("tag_list_forcheck.txt", 'wb') as f:
#     pickle.dump(tag_list, f)


# In[74]:


quesid_invertedindex={}
for i in range(len(id1)):
    if tag[i] not in quesid_invertedindex.keys():
        temp=[]
        temp.append(id1[i])
        quesid_invertedindex[tag[i]]=temp
    else:
        quesid_invertedindex[tag[i]].append(id1[i])
        


# In[75]:


flag={}
for i in quesid_invertedindex.keys():
    flag[i]=0


# In[76]:


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


# In[77]:


query_code="enter the number\nimport numpy as np\na=b+c\n"
query_body="How to add two numbers in numpy"


# In[78]:


#Assigning weights to zonal section
weight_title=0.3
weight_body=0.3
weight_code=0.4


# In[79]:


tokenizer=RegexpTokenizer('\s+',gaps=True)


# In[80]:


# preprocess(pd.DataFrame([query_body])[0])[0]


# In[81]:


body_corpus = pickle.load(open("body.txt", "rb") )
title_corpus = pickle.load(open("title.txt", "rb" ) )
tag_list = pickle.load(open("tag_list.txt", "rb" ) )


# In[82]:


# #It needed when body.txt and title.txt deleted. 

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
        


# In[ ]:


# with open("tag_list.txt", 'wb') as f:
#     pickle.dump(tag_list, f)
# with open("body.txt", 'wb') as f:
#     pickle.dump(body_corpus, f)

# with open("title.txt", 'wb') as f:
#     pickle.dump(title_corpus, f) 


# In[83]:


title_corpus_main=[]
title_corpus_main.append(title_corpus)


# In[84]:


body_corpus_main=[]
body_corpus_main.append(body_corpus)


# In[85]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[86]:


def pre(code):
    s=code.split('\n')
    
    
    t=copy.deepcopy(s)
    counter=0
    for i in range(len(t)):
        if s[counter]=="" or s[counter] =='' or s[counter]==' ':
            s.pop(counter)
        else:
            counter+=1
    return '\n'.join(s)
        


# In[87]:


def calculate_score_code(code1,code2):
    if code1=='' or code1==" " or code2=='' or code2==" " or code1==' ' or code2==' ':
        return 0
    sim1=matchSlidingWindow(code1, code2)
    
   
    
    sim2=graph_similarity_measure(adjacency_create(get_all_edges(pre(code1).split('\n'))),adjacency_create(get_all_edges(pre(code2).split('\n'))))
    score=0.4*sim1+0.6*sim2
    return score


# In[88]:


# #For body title of tagslist in which I have the body, tiltle and code for each tags at one place.
# tokenizer = RegexpTokenizer(r'\w+')
# similarity_body=[]
# similarity_title=[]
# similarity_code=[]
# tag_similar=[]
# counter=0
# for i in tag_list.keys():
#     counter+=1
#     if counter%2000==0:
#         print("Counter : ",counter)
#     a=[]
#     a.append(tag_list[i]['body'])
#     sim1=vc_body_tag.transform(a)
#     similarity_body.append(pairwise_distances(sim1, q_title1)[0][0])
#     a=[]
#     a.append(tag_list[i]['title'])
#     sim1=vc_title_tag.transform(a)
#     similarity_title.append(pairwise_distances(sim1, q_title)[0][0])    
#     tag_similar.append(i)
# #     sim1=matchCode(tag_list[i]['code'],query_code)


# # Answers Retrieval

# In[89]:


question_answers=pickle.load(open("pfiltered_ans.pickled","rb"))


# In[90]:


df2= pd.DataFrame(question_answers)
print(df2.head())


# In[91]:


#Convert everything into lists
processed=copy.deepcopy(df2)
id4=processed[0]
body1=processed[3]
code1=processed[4]
qid=processed[1]
score1=processed[2]
score1=list(score1)
id4=list(id4)
body1=list(body1)
qid=list(qid)
code1=list(code1)


# In[92]:


body_corpus1 = pickle.load(open("body_list_answers.txt", "rb") )


# In[ ]:


# # #It needed when body.txt and title.txt deleted. 
# #For answer body
# body_corpus1=''
# counter=0
# for i in range(len(body1)):
#     counter+=1
#     if counter%20000==0:
#         print("Counter : ",counter)

#     temp1=tokenizer.tokenize(body1[i])
 
#     for j in range(len(temp1)):
#         body_corpus1+=" "+ temp1[j]
        


# In[67]:


# with open("body_list_answers.txt", 'wb') as f:
#     pickle.dump(body_corpus1, f)


# In[93]:


answers_with_hash={}
for i in range(len(id4)):
    temp={}
    temp['body']=body1[i]
    temp['code']=code1[i]
    answers_with_hash[id4[i]]=temp


# In[94]:


# df2[df2[1].isin([469])].drop_duplicates(subset=[0], keep='last')


# In[95]:


body_corpus_main=[]
body_corpus_main.append(body_corpus1)


# In[108]:


def search_query(query_body,query_code):
    #To get the questions id with token of query body matched with tags
    query_body1=preprocess_for_query(query_body)
    question_id=[]
    for i in range(len(query_body1)):
        if  query_body1[i]=="python"  or query_body1[i]=='python-2.7' or query_body1[i]=='python-3.0':
            question_id=[]
            break
        if query_body1[i] in unique_list:
            question_id+=quesid_invertedindex[query_body1[i]]

    if len(question_id)==0:
        question_id+=id3
    title_corpus_main=[]
    title_corpus_main.append(title_corpus)
    body_corpus_main=[]
    body_corpus_main.append(body_corpus)
    #For Title using question body
    vc_title_tag = TfidfVectorizer() #TfidfVectorizer()
    vec_title = vc_title_tag.fit_transform(title_corpus_main)
    q_title = vc_title_tag.transform(preprocess(pd.DataFrame([query_body])[0]))

        #For body using question body
    vc_body_tag = TfidfVectorizer() #TfidfVectorizer()
    vec_body = vc_body_tag.fit_transform(body_corpus_main)
    q_title1 = vc_body_tag.transform(preprocess(pd.DataFrame([query_body])[0]))
    #For body title of tagslist in which I have the body, tiltle and code for each tags at one place.
    tokenizer = RegexpTokenizer(r'\w+')
    similarity_body=[]
    similarity_title=[]
    similarity_code=[]
    tag_similar=[]
    code_similar=[]
    counter=0
    print("Now processing the query and questions similarity....................")
    for i in question_id:
        counter+=1
    #     print("askf")
        if counter%2000==0:
            print("Counter : ",counter)
        a=[]
        a.append(questions_with_hash[i]['body'])
        sim1=vc_body_tag.transform(a)
        similarity_body.append(cosine_similarity(sim1, q_title1)[0][0])
        a=[]
        a.append(questions_with_hash[i]['title'])
        sim1=vc_title_tag.transform(a)
        similarity_title.append(cosine_similarity(sim1, q_title)[0][0])    
        tag_similar.append(i)
        try:
            sim1=calculate_score_code((' '.join(questions_with_hash[i]['code'])),query_code)
        except:
            print((' '.join(questions_with_hash[i]['code'])))
        similarity_code.append(sim1)
    #
    rel=int(input("Enter the number of relevant Questions : "))
    main_score=[]
    for i in range(len(similarity_body)):
        main_score.append(similarity_body[i]*weight_body+similarity_title[i]*weight_title+similarity_code[i]*weight_code)
    a=copy.deepcopy(np.array(main_score))
#Another approach to take top relevant tags

    indices_h=heapq.nlargest(rel, range(len(a)), a.take)

    rel_question=[]
    for i in range(rel):
        rel_question.append(id3[id3.index(question_id[indices_h[i]])])
    #Assigning weights to zonal section
    weight_body1=0.3
    weight_code1=0.3
    weight_question1=0.4
    body_corpus_main=[]
    body_corpus_main.append(body_corpus1)
    #For body using question body
    vc_body_tag = TfidfVectorizer() #TfidfVectorizer()
    vec_body = vc_body_tag.fit_transform(body_corpus_main)
    q_title1 = vc_body_tag.transform(preprocess(pd.DataFrame([query_body])[0]))
    #For body title of tagslist in which I have the body, tiltle and code for each tags at one place.
    answer_id=[]
    check_for_q=[]
    for i in range(len(rel_question)):
        pd1=df2[df2[1].isin([rel_question[i]])]
        answer_id+=list(pd1[0])
        check_for_q+=list(pd1[1])
    
    
    tokenizer = RegexpTokenizer(r'\w+')
    similarity_body=[]
    similarity_code=[]
    tag_similar=[]
    code_similar=[]
    counter=0
    print("Now processing the answers with query .....................")
    for i in answer_id:
        counter+=1
    #     print("askf")
        if counter%2000==0:
            print("Counter : ",counter)
        b=[]
        b.append(answers_with_hash[i]['body'])
        sim1=vc_body_tag.transform(b)
        similarity_body.append(cosine_similarity(sim1, q_title1)[0][0])

        tag_similar.append(i)
        sim1=calculate_score_code((' '.join(answers_with_hash[i]['code'])),query_code)
        similarity_code.append(sim1)
    #rel=int(input("Enter the number of relevant Answers : "))
    rel=int(input("Enter the number of relevant Answers : "))
    main_score=[]
    for i in range(len(similarity_body)):
        value=a[rel_question.index(check_for_q[i])]
        main_score.append(similarity_body[i]*weight_body1 + similarity_code[i]*weight_code1+ weight_question1*value)
    d=copy.deepcopy(np.array(main_score))

    #Another approach to take top relevant tags
    indices_h1=heapq.nlargest(rel, range(len(d)), d.take)

    rel_answers=[]
    for i in range(rel):
        rel_answers.append(id4[id4.index(answer_id[indices_h1[i]])])
    # print("Answers are : ")
    answers=[]
    for i in range(rel):
        answers.append(df2[df2[0].isin([rel_answers[i]])])
    return answers
    pass


# In[185]:


print("...........................User Query..........................")
query_body=input("Enter the Question Body : ")
query_code=input("Enter the Question Code")


# In[ ]:


answers=search_query(query_body,query_code)


# In[ ]:


for idx, i in enumerate(answers):
    print("Answer", idx+1)
    print(i.values[0][3])
    print("\nCode")
    try:
        print("\n".join(i.values[0][4][0].split("\n")))
    except:
        pass
    print("*"*140)
    print("\n")


# In[178]:


list(df.head()[2])


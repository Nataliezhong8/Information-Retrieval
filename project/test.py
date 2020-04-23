# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:30:46 2019

@author: lenovo
"""

import pickle
import project_part1 as project_part1


documents = {1:"Trump, Donald Trump, Donald Trump."}


#with open('test_500docs.pickle', 'rb') as handle:
#     content= pickle.load(handle)

#documents = {}
#for i in range(5):
#    documents[i] = content[i]

index = project_part1.InvertedIndex()
index.index_documents(documents)
print('tf_tokens')
print(index.tf_tokens)
print()
print('tf_entities')
print(index.tf_entities)
print()
print('idf_tokens')
print(index.idf_tokens)
print()
print('idf_entities')
print(index.idf_entities)
'''
input('please enter')
Q = 'Los The Angeles Boston Times Globe Washington Post'
DoE = {'Los Angeles Times':0, 'The Boston Globe':1,'The Washington Post':2, 'Star Tribune':3, 'Times':4, 'Boston':5}

query_splits = index.split_query(Q, DoE)

print('Possible query splits:\n')
    
input('please enter')

doc_id = 1

print('Score for each query split:\n')
result = index.max_score_query(query_splits, doc_id)

input('please enter')
print('The maximum score:\n')
print(result)'''

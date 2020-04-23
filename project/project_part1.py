'''
Author:Yinghong Zhong(z5233608)
Date: 1/11/2019(T3)

Project 1 part 1(COMP6714)
'''

import spacy
from collections import Counter
from math import log
from copy import deepcopy
from itertools import combinations

class InvertedIndex:
    

    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = []
        self.tf_entities = [] 

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = []
        self.idf_entities = []
        self.nlp = spacy.load('en_core_web_sm') #load the English module

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        
        #1. construct the term frequency for tokens and entities
        token_dict = {}
        ent_dict = {}

        for i in documents.keys():
            doc = self.nlp(documents[i])
            tokenlist, entlist = [], []

            for token in doc:
                if ((not token.is_stop) and (not token.is_punct)):
                    tokenlist.append(token.text)
            for ent in doc.ents:
                entlist.append(ent.text) 

            tcount, ecount = Counter(tokenlist),Counter(entlist) #get the count of every token

            single_count = Counter()
            #aggregate entities into a dict, key:entity, value: {docID:count}
            for key in ecount.keys(): 
                if(len(key.split()) == 1): #collect the single-word entity
                    single_count[key] = ecount[key]
                if key not in ent_dict:
                    ent_dict[key] = {i: ecount[key]}
                else:
                    ent_dict[key][i] = ecount[key]


            #aggregate tokens into a dict, key:token, value: {docID:count}
            for key in tcount.keys():
                if key in single_count.keys():  #exclude the single word entity in the token list
                    tcount[key] -= single_count[key]

                if tcount[key] != 0: 
                    if key not in token_dict:
                        token_dict[key] = {i: tcount[key]}
                    else:
                        token_dict[key][i] = tcount[key]
       
        #2. construct the tf and idf index
        tf_tokens = deepcopy(token_dict) #use the same data format 
        tf_entities = deepcopy(ent_dict)
        idf_tokens, idf_entities = {},{}
        docs_num = len(documents)

        # caluculate tf(norm) for tokens
        for token in tf_tokens.keys():
            for docid in tf_tokens[token].keys():
                tf_tokens[token][docid] = 1.0 + log(1.0 + log(tf_tokens[token][docid]))

        #calculate tf(norm) for entities
        for ent in tf_entities.keys():
            for docid in tf_entities[ent].keys():
                tf_entities[ent][docid] = 1.0 + log(tf_entities[ent][docid])

        # caluculate idf for tokens
        for token in token_dict.keys():
            idf_tokens[token] = log(docs_num / (1.0 + len(token_dict[token]))) + 1.0

        # caluculate idf for entities
        for ent in ent_dict.keys():
            idf_entities[ent] = log(docs_num / (1.0 + len(ent_dict[ent]))) + 1.0
        
        self.tf_tokens = tf_tokens
        self.tf_entities = tf_entities
        self.idf_tokens = idf_tokens
        self.idf_entities = idf_entities


    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        query_splits = []
        
        #1.select the probable entities
        '''1)a empty entity list
           2)find all combinations as eneities of the query
           3)if a entity is in DOE, append it in the entity list
        '''
        entitylist = []
        #1.1 find the single-word entities and the tokens
        tokenlist = Q.split(' ')

        for token in tokenlist:
            if (token in DoE) and (token not in entitylist):
                entitylist.append(token) #this is the single-word entity
        
        #append the split with all words as tokens and no words as entity in the query_splits dict
        query_splits.append({'tokens': tokenlist, 'entities': []})
        
        #1.2 construct combinations of tokens and find multi-word entities
        for token_nb in range(2, len(tokenlist)+1): #from length = 2 to len(tokenlist)
            for combi in combinations(tokenlist, token_nb): #get combinations
                ent = ''
                for j in range(token_nb):
                    if (j != (token_nb-1)):
                        ent += combi[j] + ' '
                    else:
                        ent += combi[j]
                if (ent in DoE) and (ent not in entitylist):
                    entitylist.append(ent) #this is the multi-word entity
        
        #2.enumerate all possible subsets of entities, using combinations
        '''
        2.1 find the combinations
        2.2 check token count of each combination doesn't exceed Q_count
        '''
        Q_count = Counter(tokenlist) #token count in Query
 
        #2.1 find the combinations
        for subset_len in range(1, len(entitylist)+1):
            for subset in combinations(entitylist, subset_len):
                #2.2.1 count the token in subset
                subset_word = []
                for j in range(subset_len): #split the words in each entity of the subset
                    subset_word += subset[j].split(' ')
                subset_counter = Counter(subset_word)
          
                #2.2.2 check that if the count of subset exceed Q-count
                exceed = 0
                for key in subset_counter.keys():
                    if subset_counter[key] > Q_count[key]:
                        exceed = 1
                        break
                        
                if exceed == 0:
                    #3 for filtered entity subset,append it to the query_splits list
                    token_counter = Q_count - subset_counter
                    querysplit = {'tokens': list(token_counter.elements()), 'entities': list(subset)}
                    if querysplit not in query_splits:
                        query_splits.append(querysplit)

        return query_splits

    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
        score_list = []
        
        for query in query_splits:
            score,score1,score2 = 0.0, 0.0, 0.0
            for ent in query['entities']:
                if (ent in self.tf_entities) and (doc_id  in self.tf_entities[ent]): # if the token appear in this doc
                    score1 += self.tf_entities[ent][doc_id] * self.idf_entities[ent] 
            for token in query['tokens']:
                if (token in self.tf_tokens) and (doc_id  in self.tf_tokens[token]): # if the token appear in this doc
                    score2 += self.tf_tokens[token][doc_id] * self.idf_tokens[token]  
            score = score1 + 0.4*score2
            score_list.append(score)
        
        max_score = max(score_list)
        return (max_score, query_splits[score_list.index(max_score)])        

"""
author: Yinghong Zhong(z5233608)
date: 2019/11/18(T3)
version: v8.0

COMP6714 projec_part2

"""

import numpy as np
import xgboost as xgb

import spacy
from collections import Counter
from math import log
from copy import deepcopy

class InvertedIndex:
    def __init__(self):
        self.tf_tokens = {}
        self.idf_tokens = {}
        self.nlp = spacy.load('en_core_web_sm')

    def index_documents(self, documents):
        
        #1. construct the term frequency for tokens and entities
        token_dict = {}
 
        for i in documents.keys():
            doc = self.nlp(documents[i])
            tokenlist = []

            for token in doc:
                if ((not token.is_stop) and (not token.is_punct)):
                    tokenlist.append(token.text)

            tcount = Counter(tokenlist) #get the count of every token

            #aggregate tokens into a dict, key:token, value: {docID:count}
            for key in tcount.keys():              
                if tcount[key] != 0: 
                    if key not in token_dict:
                        token_dict[key] = {i: tcount[key]}
                    else:
                        token_dict[key][i] = tcount[key]
       
        #2. construct the tf and idf index
        tf_tokens = deepcopy(token_dict) #use the same data format 
        idf_tokens = {}
        docs_num = len(documents)

        # caluculate tf(norm) for tokens
        for token in tf_tokens.keys():
            for docid in tf_tokens[token].keys():
                tf_tokens[token][docid] = 1.0 + log(1.0 + log(tf_tokens[token][docid]))

        # caluculate idf for tokens
        for token in token_dict.keys():
            idf_tokens[token] = log(docs_num / (1.0 + len(token_dict[token]))) + 1.0

        
        self.tf_tokens = tf_tokens
        self.idf_tokens = idf_tokens
        
class wiki_InvertedIndex():
    
     def __init__(self):
        self.tf_tokens = {}
        self.idf_tokens = {}

     def index_wikipage(self, wikipage):
        
        #1. construct the term frequency for tokens and entities
        token_dict = {}
 
        for entity_name in wikipage.keys():
            description_list = []

            for description in wikipage[entity_name]:
                description_list.append(description[2])
        
            description_count = Counter(description_list) #get the count of every token

            #aggregate tokens into a dict, key:token_lemma, value: {entity_name:count}
            for key in description_count.keys():
                if key not in token_dict:
                    token_dict[key] = {entity_name: description_count[key]}
                else:
                    token_dict[key][entity_name] = description_count[key]
       
        #2. construct the tf and idf index
        tf_tokens = deepcopy(token_dict) #use the same data format 
        idf_tokens = {}
        entity_num = len(wikipage)

        # caluculate tf(norm) for tokens, key:(token_lemma, token_pos_tag), value: {entity_name:tf-score}
        for token in tf_tokens.keys():
            for entity_name in tf_tokens[token].keys():
                tf_tokens[token][entity_name] = 1.0 + log(1.0 + log(tf_tokens[token][entity_name]))

        # caluculate idf for tokens, key:(token_lemma, token_pos_tag), value: idf-score
        for token in token_dict.keys():
            idf_tokens[token] = log(entity_num / (1.0 + len(token_dict[token]))) + 1.0
        
        self.tf_tokens = tf_tokens
        self.idf_tokens = idf_tokens

#format: {doc_title:{(offset, length):(s_idx, e_idx)}}     
#men_docs = {doc_title: text}
#train_mentions = {mention_id: {doc_title: xxx, mention: xxx, 
#                offset:xxx, length:xxx, candidate_entities:[xxx]}}
class LocalIndex():
    def __init__(self):
        self.sectionOffset = {}
    
    #dataset = train_mentions, dev_mentions
    def section_index(self, men_doc, dataset):
        for data in dataset:
            for info in data.values():
                doc_title, m_offset, length = info['doc_title'], info['offset'], info['length']
                s_idx, e_idx = 0, 0
                #find the index of the section
                for i in range(m_offset, 0, -1):
                    if men_doc[doc_title][i] == '\n':
                        s_idx = i
                        break
                for j in range(m_offset+length, len(men_doc[doc_title])):
                    if men_doc[doc_title][j] == '\n':
                        e_idx = j
                
                if doc_title in self.sectionOffset:
                    self.sectionOffset[doc_title][(m_offset, length)] = (s_idx, e_idx)
                else:
                    self.sectionOffset[doc_title] = {(m_offset, length):(s_idx, e_idx)}
        
'''
return a mention_candidate_pair, 
format for pair data set:[mention_id, doc_title, mention, candidate, offset, length], 
label: 0--this candidate is not true, 1--true candidate
'''
def mention_candidate_pair(data, labels = None):
    ret = []
    groups = []
    ret_label = []
    
    for mention_id in data.keys():
        if labels != None:
            label = labels[mention_id]['label']
        else:
            label = None
            
        groups.append(len(data[mention_id]['candidate_entities'])) #mark down the group number
        doc_title, mention = data[mention_id]['doc_title'], data[mention_id]['mention']
        #construct metion_candidate pair
        for candidate in data[mention_id]['candidate_entities']:
            pair = [mention_id, doc_title, mention, candidate, data[mention_id]['offset'], data[mention_id]['length']]
            ret.append(pair)
            if label != None:
                if candidate == label:
                    ret_label.append(1)
                else:
                    ret_label.append(0)
            
    return np.array(ret), np.array(groups), np.array(ret_label)

#raw_data = [mention_id, doc_title, mention, candidate, offset, length]       
def generate_features(raw_data, men_doc_index, wikipage_index, parsed_entity_pages, local_sectionOffset, men_docs):
    features = []
    
    mention_section = {}
    for row in raw_data:
        row_feature = [] #create a row for each mention_candidate pair
        
        mention_token = []       
        mention = men_doc_index.nlp(str(row[2]))
        doc_title, candidate = str(row[1]), str(row[3])
        mention_offset = int(row[4])
        length = int(row[5])
        #doc_title, candidate, offset, length = str(row[1]), str(row[3]), int(row(4)), int(row(5))
        
        #f0.tf-idf(tokens) of mention in wikipage
        #f_mention. tf-idf(tokens) of mention in men_doc
        for token in mention:
            if (not token.is_stop) and (not token.is_punct):
                mention_token.append(token.text)
        score0, score_mention = 0.0,0.0
        for k in mention_token:
            if (k in wikipage_index.tf_tokens) and (candidate in wikipage_index.tf_tokens[k]):
                score0 += wikipage_index.tf_tokens[k][candidate] * wikipage_index.idf_tokens[k]
            if (k in men_doc_index.tf_tokens) and (doc_title in men_doc_index.tf_tokens[k]):
                score_mention += men_doc_index.tf_tokens[k][doc_title] * men_doc_index.idf_tokens[k]
        row_feature.append(score0)
        
        words = candidate.split('_')
        can_str = ' '
        candi = can_str.join(words)
        
        #f1. find precision and recall between candidate_entity and mention
        tp = 0
        if candi.lower() == str(row[2]).lower():
            F1_score = 1
        else:
            men_split = [w.lower() for w in str(row[2]).split(' ')]
            for word in words:
                if word.lower() in men_split:
                    tp += 1
            if tp == 0:
                F1_score = 0
            else:
                precision = tp / len(words)
                recall = tp / len(men_split)
                F1_score = 2 * precision * recall /(precision + recall)
        row_feature.append(F1_score)
        
        #f2. tf-idf(token) of candidate_entity in men_doc
        score2 = 0.0
        for word in words:
            if (word in men_doc_index.tf_tokens) and (doc_title in men_doc_index.tf_tokens[word]):
                score2 += men_doc_index.tf_tokens[word][doc_title] * men_doc_index.idf_tokens[word]
        row_feature.append(score2)
        row_feature.append(score_mention-score2) #f3.diff between f_mention and f_2
        
        #f4. tf-idf(token) of candidate_entity description in men_doc
        score4 = 0.0
        #find the lemma of words in wiki page
        wiki_word = np.array(parsed_entity_pages[candidate])[:,2].tolist()
        word_counter = Counter(wiki_word)
        
        for token in word_counter:
            if (token in men_doc_index.tf_tokens) and (doc_title in men_doc_index.tf_tokens[token]):
                score4 += men_doc_index.tf_tokens[token][doc_title] * men_doc_index.idf_tokens[token] * word_counter[token]
        
        row_feature.append(score4)
        
        #prepare for the local index
        if (doc_title, mention_offset, length) in mention_section:
            local_doc = mention_section[(doc_title, mention_offset, length)]
        else:
            #get the s_idx and e_idx
            offset_index = local_sectionOffset.sectionOffset[doc_title][(mention_offset, length)]
            d_token =  men_docs[doc_title][offset_index[0]: offset_index[1]].split()
            mention_section = {(doc_title, mention_offset, length): [w.lower() for w in d_token]}
            local_doc = mention_section[(doc_title, mention_offset, length)]
        #f5 F1_local of candidate entity(token) in local section  
        tp_local = 0
        for word in words:
            if word.lower() in local_doc:
                tp_local += 1
        if tp_local == 0:
                F1_local_score = 0
        else:
            precision2 = tp_local / len(words)
            recall2 = tp_local / len(local_doc)
            F1_local_score = 2 * precision2 * recall2 /(precision2 + recall2)
        row_feature.append(F1_local_score)
        '''
        #f6 compare pos tag list between mention and candidate entity
        mention_pos = [t.pos_ for t in mention]
        candi_token = men_doc_index.nlp(candi)
        candi_pos = [t.pos_ for t in candi_token]
        if (mention_pos.sort() == candi_pos.sort()):
            row_feature.append(1)
        else:
            row_feature.append(0)'''
            
        features.append(row_feature)
        
    print(f'length of f is: ', len(features[0]))
    return np.array(features)
        
def transform_data(features, groups, labels=None):
    xgb_data = xgb.DMatrix(data=features, label=labels)
    xgb_data.set_group(groups)
    return xgb_data
    
#return result = {key: label} key is the unique mention id
def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    result = {}
    
    #step1. construct inverted index for men_doc
    men_doc_index = InvertedIndex()
    men_doc_index.index_documents(men_docs)
    
    local_sectionOffset = LocalIndex()
    local_sectionOffset.section_index(men_docs, [train_mentions, dev_mentions])
    
    #step2. construct inverted index for parsed_entity_pages
    wikipage_index = wiki_InvertedIndex()
    wikipage_index.index_wikipage(parsed_entity_pages)
   
    #step3. construct trainging set mention_candidate pair
    train_raw_data, train_groups, train_raw_label = mention_candidate_pair(train_mentions, train_labels)
    
    
    #step4. generate training feature
    train_feature = generate_features(train_raw_data, men_doc_index, wikipage_index, parsed_entity_pages, local_sectionOffset, men_docs)
    
    #step5. transform training data
    xgboost_train = transform_data(train_feature, train_groups, train_raw_label)
    
    #step6.7  construct test set mention_candidate pair and generate test feature
    test_raw_data, test_groups, test_raw_label = mention_candidate_pair(dev_mentions)
    
    test_feature = generate_features(test_raw_data, men_doc_index, wikipage_index, parsed_entity_pages, local_sectionOffset, men_docs)
    
    xgboost_test = transform_data(test_feature, test_groups)
    
    #step8. Model Training + Prediction
    param = {'max_depth': 8, 'eta': 0.05, 'silent': 1, 'objective': 'rank:pairwise',
         'min_child_weight': 0.02, 'lambda':100, 'subsample': 0.8}

    # Train the classifier...
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)

    # Predict test data...
    preds = classifier.predict(xgboost_test).tolist()
    
    #step9. Prediction scores of Each Testing Group
    idx = 0
    for group in test_groups:
        group_value = preds[idx: idx+group]
        max_value_index = group_value.index(max(group_value)) + idx #find the max score in each group
        result[int(test_raw_data[max_value_index][0])] = str(test_raw_data[max_value_index][3]) #key:mention_id
        idx += group 
        
    return result
    

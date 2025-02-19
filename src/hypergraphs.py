import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from src.consensus import Consensus, LLM_Heterogeneous_Graph_Information
import json

class Hypergraph(object):
    def __init__(self, cause_col, effect_col, dataframe, model):
        self.hypergraph = nx.DiGraph()
        self.cause_col = cause_col
        self.effect_col = effect_col
        self.dataframe = dataframe
        self.embedding_model = model
        self.dataframe[cause_col] = self.dataframe[cause_col].astype(str)
        self.dataframe[effect_col] = self.dataframe[effect_col].astype(str)

    def add_main_edges(self):
        for _, row in self.dataframe.iterrows():
            self.hypergraph.add_edge('event:' + row[self.cause_col], 'relation: First event - ' + row[self.cause_col] + ' || Second event - ' + row[self.effect_col])
            self.hypergraph.add_edge('relation: First event - ' + row[self.cause_col] + ' || Second event - ' + row[self.effect_col], 'event:' + row[self.effect_col])

    def add_main_node_labels(self):
        for _, row in self.dataframe.iterrows():
            self.hypergraph.nodes['event:' + row[self.cause_col]]['label'] = 'aux'
            self.hypergraph.nodes['event:' + row[self.effect_col]]['label'] = 'aux'
            self.hypergraph.nodes['relation: First event - ' + row[self.cause_col] + ' || Second event - ' + row[self.effect_col]]['label'] = row['Label']
    
    def add_main_node_embeddings(self):
        for node in self.hypergraph.nodes():
            self.hypergraph.nodes[node]['embedding'] = np.asarray(self.embedding_model.encode(node), dtype=np.float64)
    
    def generate_kfold_graphs(self):
        df_egae = pd.DataFrame()
        df_egae['y'] = [self.hypergraph.nodes[node]['label'] for node in self.hypergraph.nodes()]
        df_egae['node'] = [node for node in self.hypergraph.nodes()]

        nodes_aux = df_egae[df_egae['y'] == 'aux']['node'].to_list()
        df = df_egae[df_egae['y'] != 'aux'].reset_index(drop=True)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        graphs_kfold = []
        for train_index, test_index in kf.split(df['node'], df['y']):

            g_aux = self.hypergraph.copy()
            nodes_train = df[(df.index.isin(train_index)) & (df['y'] == 'causal')]['node'].to_list()
            nodes_out = df[(df.index.isin(train_index)) & (df['y'] != 'causal')]['node'].to_list()
            nodes_test = df[df.index.isin(test_index)]['node'].to_list()
            
            dict_train_test = {'train': [nodes_train,[1,0,0]],
                               'test':  [nodes_test, [0,1,0]],
                               'out':   [nodes_out,  [0,0,1]],
                               'aux':   [nodes_aux,  [0,0,1]]}

            for key in dict_train_test.keys():
                for node in dict_train_test[key][0]:
                    g_aux.nodes[node]['train'] = dict_train_test[key][1][0]
                    g_aux.nodes[node]['test'] = dict_train_test[key][1][1]
                    g_aux.nodes[node]['aux'] = dict_train_test[key][1][2]

            graphs_kfold.append(g_aux)
        return graphs_kfold

    def _generate_node_to_index(self):
        index = 0
        node_to_index = {}
        for node in self.hypergraph.nodes():
            node_to_index[node] = index
            index+=1 

class HeterogeneousHyperGraph(Hypergraph):
    def __init__(self, cause_col, effect_col, dataframe, model, dic_who, dic_when, dic_where, dataset):
        super().__init__(cause_col, effect_col, dataframe, model) 
        self.dataset_name = dataset
        self.dic_who = dic_who
        self.dic_when = dic_when
        self.dic_where = dic_where

    def return_node_types(self):
        l = []
        for node in self.hypergraph.nodes():
            l.append(self.hypergraph.nodes[node]['node_type'])
        return l
    
    def add_secundary_edges(self):
        for _, row in self.dataframe.iterrows():
            if '-1' not in row['Topic_Cause']: 
                self.hypergraph.add_edge('event:' + row[self.cause_col], 'topic:' + row['Topic_Cause'])
                self.hypergraph.add_edge('topic:' + row['Topic_Cause'], 'event:' + row[self.cause_col])
            if '-1' not in row['Topic_Effect']: 
                self.hypergraph.add_edge('event:' + row[self.effect_col], 'topic:' + row['Topic_Effect'])
                self.hypergraph.add_edge('topic:' + row['Topic_Effect'],'event:' + row[self.effect_col])
            try:
                self.hypergraph.add_edge('event:' + row[self.cause_col], 'who:' + self.dic_who[eval(row['5w1h_cause'])['Who']])
                self.hypergraph.add_edge('who:' + self.dic_who[eval(row['5w1h_cause'])['Who']], 'event:' + row[self.cause_col])
            except: g = 1
            try:
                self.hypergraph.add_edge('event:' + row[self.cause_col], 'where:' + self.dic_where[eval(row['5w1h_cause'])['Where']])
                self.hypergraph.add_edge('where:' + self.dic_where[eval(row['5w1h_cause'])['Where']], 'event:' + row[self.cause_col])
            except: g = 1
            try:
                wcs = self.dic_when[eval(row['5w1h_cause'])['When']]
                for wc in wcs:
                    self.hypergraph.add_edge('event:' + row[self.cause_col], 'when:' + wc)
                    self.hypergraph.add_edge('when:' + wc, 'event:' + row[self.cause_col])
            except: g = 1            
            try:
                self.hypergraph.add_edge('event:' + row[self.effect_col], 'who:' + self.dic_who[eval(row['5w1h_effect'])['Who']])
                self.hypergraph.add_edge('who:' + self.dic_who[eval(row['5w1h_effect'])['Who']],'event:' + row[self.effect_col])
            except: g = 1
            try:
                self.hypergraph.add_edge('event:' + row[self.effect_col], 'where:' + self.dic_where[eval(row['5w1h_effect'])['Where']])
                self.hypergraph.add_edge('where:' + self.dic_where[eval(row['5w1h_effect'])['Where']],'event:' + row[self.effect_col])
            except: g = 1
            try:
                wes = self.dic_when[eval(row['5w1h_effect'])['When']]
                for we in wes:
                    self.hypergraph.add_edge('event:' + row[self.effect_col], 'when:' + we)
                    self.hypergraph.add_edge('when:' + we,'event:' + row[self.effect_col])
            except: g = 1
    
    def _get_most_similar_embedding(self, emb, embeddings, ids):
        menor = 1
        for i in range(len(embeddings)):
            if i in ids: continue
            c = cosine(emb, embeddings[i])
            if c < menor:
                menor = c
                index = i
        return index    
    
    def _aux_pseudo_label_relations(self, neighbor4, r, nr, g_obs, con):
        if 'relation:' in neighbor4:
            if g_obs.nodes[neighbor4]['train'] == 1:
                r+=5
            else:
                llm_ckasses = con.generate_consensus(neighbor4)
                if len(llm_ckasses) == 1 and llm_ckasses[0] == 'causal':
                    r+=1
                elif len(llm_ckasses) == 1 and llm_ckasses[0] == 'non_causal':
                    nr+=1
    
        return r, nr
    
    def _pseudo_label_relations(self, neighbor2, g_obs, con):
        r, nr = 0, 0
        for neighbor3 in g_obs.neighbors(neighbor2):
            if 'event:' in neighbor3:
                for neighbor4 in g_obs.neighbors(neighbor3):
                    r, nr = self._aux_pseudo_label_relations(neighbor4, r, nr, g_obs, con)
        return r, nr

    def _add_train_relation_edges(self, k, graph_fold):
        index_to_node_train, index_to_node_test = {}, {}
        count_train, count_test = 0, 0
        embeddings_relation_train, embeddings_relation_test  = [], []
        for node in graph_fold.nodes():
            if 'relation:' in node:
                if graph_fold.nodes[node]['train'] == 1:
                    embeddings_relation_train.append(graph_fold.nodes[node]['embedding'])
                    index_to_node_train[count_train] = node
                    count_train+=1
                elif graph_fold.nodes[node]['test'] == 1:
                    embeddings_relation_test.append(graph_fold.nodes[node]['embedding'])
                    index_to_node_test[count_test] = node
                    count_test+=1
        
        A = kneighbors_graph(embeddings_relation_train, n_neighbors=k, metric='cosine', mode='connectivity', include_self=False).toarray() 
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] > 0:
                    graph_fold.add_edge(index_to_node_train[i],index_to_node_train[j])
                    graph_fold.add_edge(index_to_node_train[j],index_to_node_train[i])

        return index_to_node_train, index_to_node_test, embeddings_relation_train, embeddings_relation_test

    def _connect_to_train_edges(self, i, k, graph_fold, index_to_node_test, index_to_node_train, embeddings_relation_test, embeddings_relation_train):
        l_out = [-1]
        while len(l_out) < k:
            index_most_similar = self._get_most_similar_embedding(embeddings_relation_test[i], embeddings_relation_train, l_out)
            l_out.append(index_most_similar)
            graph_fold.add_edge(index_to_node_test[i], index_to_node_train[index_most_similar])
            graph_fold.add_edge(index_to_node_train[index_most_similar],index_to_node_test[i])        

    def _cennect_to_test_edges(self, i, k, graph_fold, index_to_node_test, embeddings_relation_test, con):
        l_out = [i]
        l_in = []
        while len(l_in) < k-1:
            j = self._get_most_similar_embedding(embeddings_relation_test[i], embeddings_relation_test, l_out)
            llm_classes = con[j]
            if llm_classes == 'non_causal':
                l_in.append(j)
                graph_fold.add_edge(index_to_node_test[i],index_to_node_test[j])
                graph_fold.add_edge(index_to_node_test[j],index_to_node_test[i])
            l_out.append(j)

    def add_relation_edges_llm(self, k, graph_fold, llm, system_prompt, user_prompt, final_user_prompt):
        index_to_node_train, index_to_node_test, embeddings_relation_train, embeddings_relation_test = self._add_train_relation_edges(k, graph_fold)
        llm_edges = LLM_Heterogeneous_Graph_Information(llm, self.dataset_name, graph_fold)
        real_class_consensus, classes_consensus = [], []

        for i in range(len(embeddings_relation_test)):
            relation = index_to_node_test[i]
            llm_edges.set_system_prompt(system_prompt)   
            llm_edges.set_initial_user_prompt(user_prompt)   
            llm_edges.set_final_user_prompt(final_user_prompt)
            pseudo_label = llm_edges.get_pseudo_label(relation)
            pseudo_label = pseudo_label.replace(pseudo_label[pseudo_label.find("<think>"):pseudo_label.find("</think>")+1], '').strip()
            r = pseudo_label.replace('```json', '').replace('```', '').replace('[', '').replace(']', '').replace('/think>', '')
            try:
                json_obj = json.loads(r)
                if json_obj['class'] == 'causal' or json_obj['class'] == 'non_causal': 
                    real_class_consensus.append(graph_fold.nodes[index_to_node_test[i]]['label'])
                    classes_consensus.append(json_obj['class'])
            except:
                print(r)
                print('--------------------------')
                print(graph_fold.nodes[index_to_node_test[i]]['label'])

        for i in range(len(embeddings_relation_test)):
            if classes_consensus[i] == 'causal':
                self._connect_to_train_edges(i, k, graph_fold, index_to_node_test, index_to_node_train, embeddings_relation_test, embeddings_relation_train)
            elif classes_consensus[i] == 'non_causal':
                self._cennect_to_test_edges(i, k, graph_fold, index_to_node_test, embeddings_relation_test, classes_consensus)
        
        return real_class_consensus, classes_consensus

    def add_relation_edges_ROUBO(self, k, metric, graph_fold):
        index_to_node_train, index_to_node_test_int, index_to_node_test_out = {}, {}, {}
        count_train, count_test_int, count_test_out = 0, 0, 0
        embeddings_relation_train, embeddings_relation_test_int, embeddings_relation_test_out  = [], [], []
        for node in graph_fold.nodes():
            if 'relation:' in node:
                if graph_fold.nodes[node]['train'] == 1:
                    embeddings_relation_train.append(graph_fold.nodes[node]['embedding'])
                    index_to_node_train[count_train] = node
                    count_train+=1
                elif graph_fold.nodes[node]['train'] == 0 and graph_fold.nodes[node]['label'] != 'causal':
                    embeddings_relation_test_out.append(graph_fold.nodes[node]['embedding'])
                    index_to_node_test_out[count_test_out] = node
                    count_test_out+=1
                elif graph_fold.nodes[node]['train'] == 0 and graph_fold.nodes[node]['label'] == 'causal':
                    embeddings_relation_test_int.append(graph_fold.nodes[node]['embedding'])
                    index_to_node_test_int[count_test_int] = node
                    count_test_int+=1
        A = kneighbors_graph(embeddings_relation_train, n_neighbors=k, metric=metric, mode='connectivity', include_self=False).toarray() 
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] == 1:
                    graph_fold.add_edge(index_to_node_train[i],index_to_node_train[j])
                    graph_fold.add_edge(index_to_node_train[j],index_to_node_train[i])
        for i in range(len(embeddings_relation_test_out)):
            index_most_similar = self._get_most_similar_embedding(embeddings_relation_test_out[i], embeddings_relation_test_out, [i])
            index_most_similar2 = self._get_most_similar_embedding(embeddings_relation_test_out[i], embeddings_relation_test_out, [i,index_most_similar])
            index_most_similar3 = self._get_most_similar_embedding(embeddings_relation_test_out[i], embeddings_relation_test_out, [i,index_most_similar,index_most_similar2])
            graph_fold.add_edge(index_to_node_test_out[i],index_to_node_test_out[index_most_similar])
            graph_fold.add_edge(index_to_node_test_out[index_most_similar],index_to_node_test_out[i])
            graph_fold.add_edge(index_to_node_test_out[i],index_to_node_test_out[index_most_similar2])
            graph_fold.add_edge(index_to_node_test_out[index_most_similar2],index_to_node_test_out[i])
            graph_fold.add_edge(index_to_node_test_out[i],index_to_node_test_out[index_most_similar3])
            graph_fold.add_edge(index_to_node_test_out[index_most_similar3],index_to_node_test_out[i])
        
        for i in range(len(embeddings_relation_test_int)):
            index_most_similar = self._get_most_similar_embedding(embeddings_relation_test_int[i], embeddings_relation_train, [-1])
            index_most_similar2 = self._get_most_similar_embedding(embeddings_relation_test_int[i], embeddings_relation_train, [index_most_similar])
            index_most_similar3 = self._get_most_similar_embedding(embeddings_relation_test_int[i], embeddings_relation_train, [index_most_similar,index_most_similar2])
            graph_fold.add_edge(index_to_node_test_int[i],index_to_node_train[index_most_similar])
            graph_fold.add_edge(index_to_node_train[index_most_similar],index_to_node_test_int[i])
            graph_fold.add_edge(index_to_node_test_int[i],index_to_node_train[index_most_similar2])
            graph_fold.add_edge(index_to_node_train[index_most_similar2],index_to_node_test_int[i])
            graph_fold.add_edge(index_to_node_test_int[i],index_to_node_train[index_most_similar3])
            graph_fold.add_edge(index_to_node_train[index_most_similar3],index_to_node_test_int[i])

    def add_secundary_node_labels(self):
        for node in self.hypergraph.nodes():
            if 'relation:' not in node and 'event:' not in node: self.hypergraph.nodes[node]['label'] = 'aux'
            if 'event:' in node: self.hypergraph.nodes[node]['node_type'] = 0
            if 'relation:' in node: self.hypergraph.nodes[node]['node_type'] = 1
            if 'topic:' in node: self.hypergraph.nodes[node]['node_type'] = 2
            if 'who:' in node: self.hypergraph.nodes[node]['node_type'] = 3
            if 'where:' in node: self.hypergraph.nodes[node]['node_type'] = 4
            if 'when:' in node: self.hypergraph.nodes[node]['node_type'] = 5
    
    def add_secundary_node_embeddings(self):
        for node in self.hypergraph.nodes():
            if 'relation:' not in node and 'event:' not in node:
                self.hypergraph.nodes[node]['embedding'] = np.asarray(self.embedding_model.encode(node), dtype=np.float64)
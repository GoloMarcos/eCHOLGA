import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class Hypergraph(object):
    def __init__(self, cause_col, effect_col, dataframe, model):
        self.hypergraph = nx.DiGraph()
        self.cause_col = cause_col
        self.effect_col = effect_col
        self.dataframe = dataframe
        self.embedding_model = model

        self.dataframe[cause_col] = self.dataframe[cause_col].astype(str)
        self.dataframe[effect_col] = self.dataframe[effect_col].astype(str)
        self.dataframe['Embedding_' + cause_col] = list(model.encode(self.dataframe[cause_col]))
        self.dataframe['Embedding_' + effect_col] = list(model.encode(self.dataframe[effect_col]))
    
    def add_main_edges(self):
        for _, row in self.dataframe.iterrows():
            self.hypergraph.add_edge('event:' + row[self.cause_col], 'relation: ' + row[self.cause_col] + '_' + row[self.effect_col])
            self.hypergraph.add_edge('relation: ' + row[self.cause_col] + '_' + row[self.effect_col], 'event:' + row[self.effect_col])

    def add_main_node_labels(self):
        for _, row in self.dataframe.iterrows():
            self.hypergraph.nodes['event:' + row[self.cause_col]]['label'] = 'aux'
            self.hypergraph.nodes['event:' + row[self.effect_col]]['label'] = 'aux'
            self.hypergraph.nodes['relation: ' + row[self.cause_col] + '_' + row[self.effect_col]]['label'] = row['Label']
    
    def add_main_node_embeddings(self):
        for _, row in self.dataframe.iterrows():
            cause_emb = np.asarray(row['Embedding_' + self.cause_col], dtype=np.float64)
            effect_emb = np.asarray(row['Embedding_' + self.effect_col], dtype=np.float64)
            self.hypergraph.nodes['event:' + row[self.cause_col]]['embedding'] = cause_emb
            self.hypergraph.nodes['event:' + row[self.effect_col]]['embedding'] = effect_emb
            self.hypergraph.nodes['relation: ' + row[self.cause_col] + '_' + row[self.effect_col]]['embedding'] = np.mean([cause_emb,effect_emb], axis=0)
    
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
                               'test': [nodes_test,[0,1,0]],
                               'out': [nodes_out,[0,0,1]],
                               'aux': [nodes_aux,[0,0,1]]}

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
    def __init__(self, cause_col, effect_col, dataframe, model, dic_who, dic_when, dic_where):
        super().__init__(cause_col, effect_col, dataframe, model) 

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
    
    def add_relation_edges(self):
        index_to_node = {}
        embeddings_relation = []
        count = 0
        for node in self.hypergraph.nodes():
            if 'relation:' in node:
                embeddings_relation.append(self.hypergraph.nodes[node]['embedding'])
                index_to_node[count] = node
                count+=1
        
        # to do
        # grafo knn com embeddings_relation
        # pegar as conexões e adicionar no grafo com o index_to_node

    def add_secundary_node_labels(self):
        for node in self.hypergraph.nodes():
            if 'relation:' not in node and 'event:' not in node: self.hypergraph.nodes[node]['label'] = 'aux'
            if 'event:' in node: self.hypergraph.nodes[node]['node_type'] = 0
            if 'relation:' in node: self.hypergraph.nodes[node]['node_type'] = 1
            if 'topic:' in node: self.hypergraph.nodes[node]['node_type'] = 2
            if 'who:' in node: self.hypergraph.nodes[node]['node_type'] = 3
            if 'when:' in node: self.hypergraph.nodes[node]['node_type'] = 4
            if 'where:' in node: self.hypergraph.nodes[node]['node_type'] = 5
    
    def add_secundary_node_embeddings(self):
        for node in self.hypergraph.nodes():
            if 'relation:' not in node and 'event:' not in node:
                self.hypergraph.nodes[node]['embedding'] = np.asarray(self.embedding_model.encode(node), dtype=np.float64)
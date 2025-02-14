import pandas as pd
import numpy as np

class Consensus(object):
    def __init__(self, llms: list, dataset: str):
        self.llms = llms
        self.dataset_name = dataset
        self.dfs = {}
        for key in self.llms:
            self.dfs[key] = pd.read_csv('./results/' + self.dataset_name + '_' + key + '.csv')
    
    def get_llms(self):
        return self.llms
    
    def generate_consensus(self, relation):
        llms_classes = []
        for key in self.dfs.keys():
            df_llm = self.dfs[key]
            e1 = relation.split(' || ')[0].replace('relation: First event: - ', '')
            e2 = relation.split(' || ')[1].replace('Second event: - ','')
            llms_classes.append(df_llm[(df_llm[' Cause'] == e1) & (df_llm[' Effect'] == e2)]['llm_class'].iloc[0])
        return np.unique(llms_classes)


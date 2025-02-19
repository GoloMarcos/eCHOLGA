import pandas as pd
import numpy as np
import ollama
import networkx as nx
import ollama
from ollama import chat
from ollama import ChatResponse

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
            e1 = relation.split(' || ')[0].replace('relation: First event - ', '')
            e2 = relation.split(' || ')[1].replace('Second event - ','')
            llms_classes.append(df_llm[(df_llm[' Cause'] == e1) & (df_llm[' Effect'] == e2)]['llm_class'].iloc[0])
        return np.unique(llms_classes)

class LLM_Heterogeneous_Graph_Information(object):
    def __init__(self, llm_ollama_name: list, dataset: str, het_graph: nx.DiGraph):
        self.llm_name = llm_ollama_name
        self.dataset = dataset
        self.het_graph = het_graph
        
    def get_llm_name(self):
        return self.llm_name
    
    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def set_initial_user_prompt(self, user_prompt: str):
        self.initial_user_prompt = user_prompt

    def set_final_user_prompt(self, user_prompt: str):
        self.final_user_prompt = user_prompt

    def _generate_custom_user_prompt(self, relation):
        user_prompt = self.initial_user_prompt

        e1 = relation.split(' || ')[0].replace('relation: First event - ', '')
        e2 = relation.split(' || ')[1].replace('Second event - ','')
        user_prompt += 'Event one: ' + e1 + '\nEvent two: ' + e2  + '\n\n### Graph information related to the events:\n'

        for e in [e1,e2]:
           for neighbor in self.het_graph.neighbors('event:' + e):
               for node_type in ['topic:', 'who:', 'when:', 'where:']:
                   if node_type in neighbor:
                        user_prompt += 'In the explored heterogeneous graph, the event ' + e + ' is connected with node ' + neighbor + '.\n'

        return user_prompt
    
    def get_pseudo_label(self, relation):
        response: ChatResponse = chat(model=self.llm_name, 
                                      messages=[{'role': 'system','content': self.system_prompt},
                                                {'role': 'user','content': self._generate_custom_user_prompt(relation)}],
                                      options={'temperature': 0, 'num_ctx': 10240, 'seed': 81})
        return response['message']['content'].strip()
    




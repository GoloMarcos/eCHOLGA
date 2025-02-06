from torch_geometric.data import HeteroData
import torch
import numpy as np 

class DataHetero(object):
    def __init__(self, hypergraph, device, list_node_types):
        self.hypergraph = hypergraph
        self.device = device
        self.data = HeteroData()
        self.list_node_types = list_node_types
        self.node_to_index = {}
        self._generate_index_for_all_node_types()

    def _generate_index_for_all_node_types(self):
        for node_type in self.list_node_types:
            self._generate_node_index(node_type)

    def _generate_node_index(self, node_type):
        self.node_to_index[node_type] = {}
        index = 0
        for node in self.hypergraph.nodes():
            if node_type in node:
                self.node_to_index[node_type][node] = int(index)
                index+=1

    def get_node_to_index(self):
        return self.node_to_index
    
    def return_node_types(self):
        l = []
        for node_type in self.list_node_types:
            for node in self.hypergraph.nodes():
                if node_type in node:
                    l.append(self.hypergraph.nodes[node]['node_type'])
        return l

    def get_data(self) -> HeteroData:
        return self.data.to(self.device)
    
    def add_nodes_by_type(self):
        for node_type in self.list_node_types:
            l = []
            for node in self.hypergraph.nodes():
                if node_type in node:
                    l.append(self.hypergraph.nodes[node]['embedding'])

            self.data[node_type].x = torch.Tensor(np.array(l))
    
    def add_edge_index_dict(self, list_edge_tuple):
        for edge_tuple in list_edge_tuple:
            sources = []
            targets = []
            for edge in self.hypergraph.edges():
                if edge_tuple[0] in edge[0] and edge_tuple[2] in edge[1]:
                    sources.append(self.node_to_index[edge_tuple[0]][edge[0]])
                    targets.append(self.node_to_index[edge_tuple[2]][edge[1]])

            self.data[edge_tuple].edge_index = torch.Tensor(np.array([np.array(sources, dtype=np.int64), np.array(targets, dtype=np.int64)])).type(torch.LongTensor)

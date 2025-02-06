import torch
from torch_geometric.nn import HGTConv, GCNConv
import torch.nn as nn
import numpy as np
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.models import GAE
from sklearn.metrics import classification_report

class Encoder_eCHOLGA(torch.nn.Module):
    def __init__(self, input_len, dim, num_node_types, dataHetero):
        super(Encoder_eCHOLGA, self).__init__()
        self.data = dataHetero
        self.conv1 = HGTConv(input_len, 128, self.data.get_data().metadata())
        self.conv2 = HGTConv(128, dim, self.data.get_data().metadata())
        self.encoder_layer_3 = nn.Linear(dim, num_node_types)
    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict)
        all_representations = nn.Tanh()(torch.cat([x_dict2[self.data.list_node_types[0]],x_dict2[self.data.list_node_types[1]],
                                                   x_dict2[self.data.list_node_types[2]],x_dict2[self.data.list_node_types[3]],
                                                   x_dict2[self.data.list_node_types[4]],x_dict2[self.data.list_node_types[5]]]))
        return nn.Tanh()(x_dict2['relation:']), all_representations, self.encoder_layer_3(all_representations)

class Encoder_eCOLGCN(torch.nn.Module):
    def __init__(self, input_len, dim, num_node_types, dataHetero):
        super(Encoder_eCOLGCN, self).__init__()
        self.data = dataHetero
        self.conv1 = GCNConv(input_len, 128)
        self.conv2 = GCNConv(128, dim)
        self.encoder_layer_3 = nn.Linear(dim, num_node_types)
    def forward(self, x_dict, edge_index_dict):
        x_dict1 = nn.Tanh()(self.conv1(x_dict, edge_index_dict))
        x_dict2 = nn.Tanh()(self.conv2(x_dict1, edge_index_dict))
        return x_dict2, self.encoder_layer_3(x_dict2)

class HeterogeneousGNNModel(object):
    def __init__(self, device, lr, radius_value, dim, num_node_types, dataHetero, graph, embedding_len):
        self.device = torch.device(device)
        self.learning_rate = lr
        self.radius_value = radius_value
        self.learned_dimension = dim
        self.num_node_types = num_node_types
        self.dataHetero = dataHetero
        self.graph = graph
        self.embedding_len = embedding_len
        self.graph_torch = from_networkx(self.graph).to(self.device)
        #self.model = GAE(Encoder_eCHOLGA(self.embedding_len, self.learned_dimension, self.num_node_types, self.dataHetero)).to(self.device)
        self.model = GAE(Encoder_eCOLGCN(self.embedding_len, self.learned_dimension, self.num_node_types, self.dataHetero)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.center = torch.Tensor([0] * self.learned_dimension).to(self.device)
        self.radius = torch.Tensor([self.radius_value]).to(self.device)
        self.mask, self.unsup_mask, unsup_nodes = self._one_class_masking()
        g_unsup = self.graph.subgraph(unsup_nodes)
        self.graph_torch_unsup = from_networkx(g_unsup).to(self.device)

    def _one_class_masking(self):
        train_mask = np.zeros(len(self.graph.nodes), dtype='bool')
        unsup_mask = np.zeros(len(self.graph.nodes), dtype='bool')
        normal_train_idx, unsup_idx, unsup = [], [], []
        count = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node]['train'] == 1:
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
                unsup.append(node)
            count += 1
        train_mask[normal_train_idx] = 1
        unsup_mask[unsup_idx] = 1

        return train_mask, unsup_mask, unsup
    
    def get_device(self):
        return self.device
    def get_learning_rate(self):
        return self.learning_rate
    def get_radius_value(self):
        return self.radius_value
    def get_learned_dimension(self):
        return self.learned_dimension
    def get_embedding_len(self):
        return self.embedding_len
    def get_graph_torch(self):
        return self.graph_torch
    def get_model(self):
        return self.model
    def get_optimizer(self):
        return self.optimizer
    def get_center(self):
        return self.center
    def get_radius(self):
        return self.radius
    def get_mask(self):
        return self.mask
    def get_unsup_mask(self):
        return self.unsup_mask
    def get_graph_torch_unsup(self):
        return self.graph_torch_unsup

    def one_class_homogeneousGNN_prediction(self, learned_representations, node_to_index, dic):
        with torch.no_grad():
            interest, outlier = [], []
            for node in self.graph.nodes():
                if self.graph.nodes[node]['test'] == 1 and self.graph.nodes[node]['label'] == 'causal':
                    interest.append(learned_representations[node_to_index[node]].cpu().numpy())
                elif self.graph.nodes[node]['test'] == 1 and self.graph.nodes[node]['label'] == 'non_causal':
                    outlier.append(learned_representations[node_to_index[node]].cpu().numpy())
            return self._ocl_prediction(interest, outlier, dic)
        
    def one_class_heterogeneousGNN_prediction(self, learned_representations, node_to_index, dic):
        with torch.no_grad():
            interest, outlier = [], []
            for node in self.graph.nodes():
                node_type = node.split(':')[0] + ':'
                if self.graph.nodes[node]['test'] == 1 and self.graph.nodes[node]['label'] == 'causal':
                    interest.append(learned_representations[node_type][node_to_index[node_type][node]].cpu().numpy())
                elif self.graph.nodes[node]['test'] == 1 and self.graph.nodes[node]['label'] == 'non_causal':
                    outlier.append(learned_representations[node_type][node_to_index[node_type][node]].cpu().numpy()) 
            return self._ocl_prediction(interest, outlier, dic)
    
    def _ocl_prediction(self,interest, outlier, dic):           
        dist_int = np.sum((interest - self.center.cpu().numpy()) ** 2, axis=1)
        scores_int = dist_int - self.radius.cpu().numpy() ** 2
        dist_out = np.sum((outlier - self.center.cpu().numpy()) ** 2, axis=1)
        scores_out = dist_out - self.radius.cpu().numpy() ** 2
        preds_interest = ['causal' if score < 0 else 'non_causal' for score in scores_int]
        preds_outliers = ['non_causal' if score > 0 else 'causal' for score in scores_out]
        y_true = ['causal'] * len(preds_interest) + ['non_causal'] * len(preds_outliers)
        y_pred = list(preds_interest) + list(preds_outliers)
        if dic:
            return classification_report(y_true, y_pred, output_dict=dic)
        else:
            return y_true, y_pred

def one_class_loss(center, radius, learned_representations, mask):

    scores = anomaly_score(center, radius, learned_representations, mask)

    ocl_loss = torch.mean(torch.where(scores > 0, scores + 1, torch.exp(scores)))

    return ocl_loss

def anomaly_score(center, radius, learned_representations, mask):

    l_r_mask = torch.BoolTensor(mask)

    dist = torch.sum((learned_representations[l_r_mask] - center) ** 2, dim=1)

    scores = dist - radius ** 2

    return scores

    
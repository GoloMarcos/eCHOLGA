import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import random
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GAE
import pandas as pd

class Encoder_eCOLGAT(torch.nn.Module):
    def __init__(self, input_len, dim, heads):
        super(Encoder_eCOLGAT, self).__init__()

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.encoder_layer_1 = GATv2Conv(input_len, 256, heads=heads, concat=False)
        self.encoder_layer_2 = GATv2Conv(256, dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        rep_node_l1 = nn.LeakyReLU(0.2)(self.encoder_layer_1(x=x, edge_index=edge_index)) 
        rep_node_l2 = nn.Tanh()(self.encoder_layer_2(x=rep_node_l1, edge_index=edge_index))

        return rep_node_l2
    

def one_class_loss(center, radius, learned_representations, mask):

    scores = anomaly_score(center, radius, learned_representations, mask)

    ocl_loss = torch.mean(torch.where(scores > 0, scores + 1, torch.exp(scores)))

    return ocl_loss

def anomaly_score(center, radius, learned_representations, mask):

    l_r_mask = torch.BoolTensor(mask)

    dist = torch.sum((learned_representations[l_r_mask] - center) ** 2, dim=1)

    scores = dist - radius ** 2

    return scores

def one_class_masking(G):

    train_mask = np.zeros(len(G.nodes), dtype='bool')
    unsup_mask = np.zeros(len(G.nodes), dtype='bool')

    normal_train_idx = []
    unsup_idx = []
    unsup = []
    
    count = 0
    for node in G.nodes():
        if G.nodes[node]['train'] == 1:
            normal_train_idx.append(count)
        else:
            unsup_idx.append(count)
            unsup.append(node)
        count += 1

    train_mask[normal_train_idx] = 1
    unsup_mask[unsup_idx] = 1

    return train_mask, unsup_mask, unsup

def One_Class_GNN_prediction(center, radius, node_to_index, learned_representations, G, dic=False):

    with torch.no_grad():
        for node in G.nodes:
            G.nodes[node]['embedding_colgat'] = learned_representations[node_to_index[node]]

        interest = []
        outlier = []
        for node in G.nodes:
            if G.nodes[node]['test'] == 1 and G.nodes[node]['label'] == 'causal':
                interest.append(G.nodes[node]['embedding_colgat'].cpu().numpy())
            elif G.nodes[node]['test'] == 1 and G.nodes[node]['label'] == 'non_causal':
                outlier.append(G.nodes[node]['embedding_colgat'].cpu().numpy())

        dist_int = np.sum((interest - center.cpu().numpy()) ** 2, axis=1)

        scores_int = dist_int - radius.cpu().numpy() ** 2

        dist_out = np.sum((outlier - center.cpu().numpy()) ** 2, axis=1)

        scores_out = dist_out - radius.cpu().numpy() ** 2

        preds_interest = ['causal' if score < 0 else 'non_causal' for score in scores_int]
        preds_outliers = ['non_causal' if score > 0 else 'causal' for score in scores_out]

        y_true = ['causal'] * len(preds_interest) + ['non_causal'] * len(preds_outliers)
        y_pred = list(preds_interest) + list(preds_outliers)
        if dic:
            return classification_report(y_true, y_pred, output_dict=dic)
        else:
            return y_true, y_pred

def initialize_homogeneous_model(g, lr, ra, dim, heads):

    device = torch.device('cuda:0')

    graph_torch = from_networkx(g).to(device)

    model = GAE(Encoder_eCOLGAT(384, dim, heads))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    c = [0] * dim
    center = torch.Tensor(c)
    center = center.to(device)
    r = [ra]
    radius = torch.Tensor(r)
    radius = radius.to(device)

    mask, unsup_mask, unsup_nodes = one_class_masking(g)
    g_unsup = g.subgraph(unsup_nodes)
    graph_torch_unsup = from_networkx(g_unsup).to(device)

    return model, optimizer, graph_torch, unsup_mask, graph_torch_unsup, center, radius, mask

def generate_kfold_graphs(g_first):
    df_egae = pd.DataFrame()

    df_egae['y'] = [g_first.nodes[node]['label'] for node in g_first.nodes()]
    df_egae['node'] = [node for node in g_first.nodes()]

    nodes_aux = df_egae[df_egae['y'] == 'aux']['node'].to_list()

    df = df_egae[df_egae['y'] != 'aux'].reset_index(drop=True)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    graphs_kfold = []
    for train_index, test_index in kf.split(df['node'], df['y']):

        g_aux = g_first.copy()

        nodes_train = df[(df.index.isin(train_index)) & (df['y'] == 'causal')]['node'].to_list()

        nodes_out = df[(df.index.isin(train_index)) & (df['y'] != 'causal')]['node'].to_list()
        
        nodes_test = df[df.index.isin(test_index)]['node'].to_list()
        
        for node in nodes_train:
            g_aux.nodes[node]['train'] = 1
            g_aux.nodes[node]['test'] = 0
            g_aux.nodes[node]['aux'] = 0

        for node in nodes_test:
            g_aux.nodes[node]['train'] = 0
            g_aux.nodes[node]['test'] = 1
            g_aux.nodes[node]['aux'] = 0

        for node in nodes_out:
            g_aux.nodes[node]['train'] = 0
            g_aux.nodes[node]['test'] = 0
            g_aux.nodes[node]['aux'] = 1

        for node in nodes_aux:
            g_aux.nodes[node]['train'] = 0
            g_aux.nodes[node]['test'] = 0
            g_aux.nodes[node]['aux'] = 1
    
        graphs_kfold.append(g_aux)
    return graphs_kfold
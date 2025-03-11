import torch.nn as nn
import torch 
from eCHOLGA import HeterogeneousGNNModel
from eCHOLGA import one_class_loss

class GNNTrainer(object):
    def __init__(self, device: str, epochs: int, heterogeneous_model: HeterogeneousGNNModel, 
                 initial_ocl_factor: float, initial_rec_factor: float, initial_nt_factor: float,
                 final_ocl_factor: float, final_rec_factor: float, final_nt_factor: float,
                 degree_ocl_factor: float, degree_rec_factor: float, degree_nt_factor: float,
                 learning_transformation: int):
        self.device = torch.device(device) 
        self.epochs = epochs 
        self.heterogeneous_model = heterogeneous_model
        self.initial_ocl_factor = initial_ocl_factor
        self.initial_rec_factor = initial_rec_factor
        self.initial_nt_factor = initial_nt_factor
        self.final_ocl_factor = final_ocl_factor
        self.final_rec_factor = final_rec_factor
        self.final_nt_factor = final_nt_factor
        self.degree_ocl_factor = degree_ocl_factor
        self.degree_rec_factor = degree_rec_factor
        self.degree_nt_factor = degree_nt_factor
        self.learning_transformation = learning_transformation
    
    def _train_model_one_epoch(self, relation_mask, relation_edge_index, node_type_labels, ocl_factor, rec_factor, nt_factor):
        self.heterogeneous_model.get_gnn_model().train()
        self.heterogeneous_model.get_optimizer().zero_grad()
        node_representation, pred_node_type = self.heterogeneous_model.get_gnn_model().encode(self.heterogeneous_model.get_graph_torch().embedding.float(), self.heterogeneous_model.get_graph_torch().edge_index)
        loss_ocl = one_class_loss(self.heterogeneous_model.get_center(), self.heterogeneous_model.get_radius(), node_representation, self.heterogeneous_model.get_mask())
        loss_rec = self.heterogeneous_model.get_gnn_model().recon_loss(node_representation[relation_mask],relation_edge_index)
        loss_nt = nn.CrossEntropyLoss()(pred_node_type, torch.Tensor(node_type_labels).squeeze().long().to('cuda:0'))
        loss = loss_ocl * min(self.final_ocl_factor, ocl_factor)
        if self.final_rec_factor != 0:
            loss+= loss_rec * min(self.final_rec_factor, rec_factor)
        loss+= loss_nt * max(self.final_nt_factor, nt_factor)
        loss.backward()
        self.heterogeneous_model.get_optimizer().step()
        return loss, loss_ocl, loss_rec, loss_nt, node_representation

    def train_model(self, verbose: bool):
        relation_mask = self.heterogeneous_model.return_unsup_mask('relation')
        relation_edge_index = self.heterogeneous_model.return_graph_torch_unsup('relation').edge_index
        ocl_factor, rec_factor, nt_factor = self.initial_ocl_factor, self.initial_rec_factor, self.initial_nt_factor
        node_type_labels = [self.heterogeneous_model.get_graph().nodes[node]['node_type'] for node in self.heterogeneous_model.get_graph().nodes()]
        embeddings = []

        for epoch in range(self.epochs):
            loss, loss_ocl, loss_rec, loss_nt, node_representation = self._train_model_one_epoch(relation_mask, relation_edge_index, node_type_labels, ocl_factor, rec_factor, nt_factor)
            embeddings.append(node_representation)
            if epoch%self.learning_transformation == 0:
                ocl_factor = ocl_factor + self.degree_ocl_factor
                rec_factor = rec_factor + self.degree_rec_factor
                nt_factor = max(0,(nt_factor - self.degree_nt_factor)) 
                _,_, dic_metrics = self.predict(node_representation)
                if verbose: print(f'Ep {epoch} | Total Loss: {loss.detach().cpu().numpy():.3f} | OCL Loss: {loss_ocl.detach().cpu().numpy():.3f} | Rec Loss: {loss_rec.detach().cpu().numpy():.3f} | NT Loss: {loss_nt.detach().cpu().numpy():.3f} | F1-macro: {dic_metrics['macro avg']['f1-score']*100:.2f}%')
        return embeddings

    def predict(self, node_representation):
        index, node_to_index = 0, {}
        for node in self.heterogeneous_model.get_graph().nodes():
            node_to_index[node] = index
            index+=1
        y_true, y_pred = self.heterogeneous_model.one_class_homogeneousGNN_prediction(node_representation, node_to_index, False)
        dic_metrics = self.heterogeneous_model.one_class_homogeneousGNN_prediction(node_representation, node_to_index, True)
        return y_true, y_pred, dic_metrics

    
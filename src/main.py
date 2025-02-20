import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from hypergraphs import HeterogeneousHyperGraph
from components import FiveWOneH
import components
from gnn_trainer import GNNTrainer
import torch
import random
from eCHOLGA import HeterogeneousGNNModel
from utils import plot_confusion_matrix, init_metrics, write_results, save_values
from sklearn.metrics import classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eCHOLGA')
    parser.add_argument("--dataset", type=str, default="Headlines", help="dataset name")
    parser.add_argument("--radius", type=float, default=0.35, help="hypershpere radius")
    parser.add_argument("--lr", type=float, default=0.008, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=6000, help="training epochs")
    parser.add_argument("--llm", type=str, default="qwen2.5:14b", help="llm to generate pseudo labels")
    parser.add_argument("--ocl_i", type=float, default=0.00, help="initial ocl loss factor")
    parser.add_argument("--ocl_f", type=float, default=0.50, help="final ocl loss factor")
    parser.add_argument("--ocl_d", type=float, default=0.02, help="degree ocl loss factor")
    parser.add_argument("--rec_i", type=float, default=0.00, help="initial rec loss factor")
    parser.add_argument("--rec_f", type=float, default=0.00, help="final rec loss factor")
    parser.add_argument("--rec_d", type=float, default=0.00, help="degree rec loss factor")
    parser.add_argument("--nt_i", type=float,  default=1.00, help="initial nt loss factor")
    parser.add_argument("--nt_f", type=float,  default=0.50, help="final nt loss factor")
    parser.add_argument("--nt_d", type=float,  default=0.02, help="degree nt loss factor")

    args = parser.parse_args()
    df = pd.read_csv('../datasets/' + args.dataset + '.csv')
    if args.dataset == 'Headlines':
        threew = FiveWOneH(args.dataset, df, 0.2)
        dic_who_headlines = threew.generate_dict('Who')
        dic_where_headlines = threew.generate_dict('Where')
        dic_when, dic_where, dic_who = components.dic_when_headlines, dic_where_headlines, dic_who_headlines
        num_node_types = 6
    elif args.dataset == 'Risk':
        dic_when, dic_where, dic_who = components.dic_when_risk, components.dic_where_risk, components.dic_who_risk
        num_node_types = 5
    elif args.dataset == 'FinCausal':
        dic_when, dic_where, dic_who = components.dic_when_fincausal, components.dic_where_fincausal, components.dic_who_fincausal
        num_node_types = 6
    elif args.dataset == 'Twitter':
        threew = FiveWOneH(args.dataset, df, 0.2)
        dic_who_twitter = threew.generate_dict('Who')
        dic_when, dic_where, dic_who = components.dic_when_twitter, components.dic_where_twitter, dic_who_twitter
        num_node_types = 6

    system_prompt = """You are an AI for causal reasoning designed to detect causality between events. Your task is to classify the event one causes the event two, returnin a structured JSON format with the class "causal" or "non_causal". The class needs to be the strings "causal" or "non_causal".

Your output should be formatted as a JSON object. Below is an example of the expected output structure:

```json
{
"class": "causal"
}
```"""

    user_prompt = """Please read the following event text pairs and some graph information about the events to detect causality between the events. Your response should be the class representing if the event one causes the event two, in the form of a JSON, in which, the class can be "causal" or "non_causal", with the following structure:

```json
{
"class": "non_causal"
}
```

Important: Your output must be in JSON format only. No additional text, explanations, or comments are allowed. Do not include any other information outside of the JSON structure.

Now, apply the same logic to the following event pairs.

### Event pairs to analyze:\n
# """
    m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    het_hyperG = HeterogeneousHyperGraph(' Cause', ' Effect', df, m, dic_who, dic_when, dic_where, args.dataset)
    het_hyperG.add_main_edges()
    het_hyperG.add_main_node_labels()
    het_hyperG.add_main_node_embeddings()
    het_hyperG.add_secundary_edges()
    het_hyperG.add_secundary_node_labels()
    het_hyperG.add_secundary_node_embeddings()
    graphs_kfold = het_hyperG.generate_kfold_graphs()

    path_results = '../results/' + args.dataset + '_'
    list_parameters = str(args.radius) + '_' + str(args.lr) + '_' + str(args.n_epochs) + '_' + str(args.llm) + '_' + str(args.ocl_i) + '_' + str(args.ocl_f) + '_' + str(args.ocl_d) + '_' + str(args.rec_i) + '_' + str(args.rec_f) + '_' + str(args.rec_d) + '_' + str(args.nt_i) + '_' + str(args.nt_f) + '_' + str(args.nt_d)
    metrics = init_metrics()

    for g_fold in graphs_kfold:
        het_hyperG.add_relation_edges_pseudo_labels(4, g_fold, args.llm, system_prompt, user_prompt)
        heterogeneous_model = HeterogeneousGNNModel('cuda:0', args.lr, args.radius, 3, num_node_types, g_fold, 384)
        gnn_trainer = GNNTrainer('cuda:0', args.n_epochs, heterogeneous_model, args.ocl_i, args.rec_i, args.nt_i, args.ocl_f, args.rec_f, args.nt_f, args.ocl_d, args.rec_d, args.nt_d, 100)
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        torch.cuda.manual_seed_all(42)
        embs = gnn_trainer.train_model(False)
        y_true, y_pred, dic_metrics = gnn_trainer.predict(embs[-1])
        save_values(metrics, dic_metrics)

    write_results(metrics, 'echolga.csv', list_parameters, path_results)



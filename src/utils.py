from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import numpy as np
from pathlib import Path

def show_graph(G):
  ### ARESTAS
  edge_x = []
  edge_y = []

  # adicionando as coordenadas
  for edge in G.edges():
      x0, y0 = G.nodes[edge[0]]['pos']
      x1, y1 = G.nodes[edge[1]]['pos']
      edge_x.append(x0)
      edge_x.append(x1)
      edge_x.append(None)
      edge_y.append(y0)
      edge_y.append(y1)
      edge_y.append(None)

  # definindo cor e estilo das arestas
  edge_trace = go.Scatter(
      x=edge_x, y=edge_y,
      line=dict(width=2, color='#888'),
      hoverinfo='none',
      mode='lines')

  ### VÉRTICES
  node_x = []
  node_y = []

  # adicionando as coordenadas
  #text = []
  for node in G.nodes():
      x, y = G.nodes[node]['pos']
      node_x.append(x)
      node_y.append(y)
      #text.append(G.nodes[node]['text']) 

  # definindo cor e estilo dos vértices
  node_trace = go.Scatter(
      x=node_x, y=node_y,
      mode='markers',
      hoverinfo='text',
      #hovertext= text,
      marker=dict(
          size=10,
          line_width=2))

  node_labels = []
  for node in G.nodes():
    node_labels.append(G.nodes[node]['node_type'])

  node_trace.marker.color = node_labels
  
  # visualizando!
  fig = go.Figure(data=[edge_trace, node_trace],
              layout=go.Layout(
                  showlegend=False,
                  hovermode='closest',
                  margin=dict(b=20,l=5,r=5,t=40),
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                  )
  fig.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['causal', 'non_causal'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def init_metrics():
    metrics = {
        'causal': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'non_causal': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'macro avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'weighted avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'accuracy': []
    }
    return metrics


def save_values(metrics, values):
    for key in metrics.keys():
      if key == 'accuracy':
        metrics[key].append(values[key])
      else:
        for key2 in metrics[key].keys():
          metrics[key][key2].append(values[key][key2])

def write_results(metrics, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'
        for key in metrics.keys():
            if key == 'accuracy':
              string += ';' + key + '-mean;' + key + '-std'
            else:
              for key2 in metrics[key].keys():
                string += ';' + key + '_' + key2 + '-mean;' + key + '_' + key2 + '-std'

        string += '\n'
        file_.write(string)
        file_.close()

    file_ = open(path + file_name, 'a')
    string = line_parameters

    for key in metrics.keys():
      if key == 'accuracy':
        string += ';' + str(np.mean(metrics[key])) + ';' + str(np.std(metrics[key]))
      else:
        for key2 in metrics[key].keys():
          string += ';' + str(np.mean(metrics[key][key2])) + ';' + str(np.std(metrics[key][key2]))

    string += '\n'
    file_.write(string)
    file_.close()


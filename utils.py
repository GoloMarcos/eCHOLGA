import numpy as np
import networkx as nx
import pandas as pd

def generate_homogeneous_graph(df, model, llm_class = False, cause_col = ' Cause', effect_col = ' Effect'):
   
   graph = nx.DiGraph()
   df[cause_col] = df[cause_col].astype(str)
   df[effect_col] = df[effect_col].astype(str)

   df['Embedding_' + cause_col] = list(model.encode(df[cause_col]))
   df['Embedding_' + effect_col] = list(model.encode(df[effect_col]))

   for _, row in df.iterrows():
      cause = row[cause_col]
      effect = row[effect_col]
      
      graph.add_edge('event:' + cause, 'relation: ' + cause + '_' + effect)
      graph.add_edge('relation: ' + cause + '_' + effect, 'event:' + effect)
      
      graph.nodes['event:' + cause]['label'] = 'aux'
      graph.nodes['event:' + effect]['label'] = 'aux'
      graph.nodes['relation: ' + cause + '_' + effect]['label'] = row['Label']
      if llm_class:
        graph.nodes['event:' + cause]['llm_label'] = 'aux'
        graph.nodes['event:' + effect]['llm_label'] = 'aux'
        graph.nodes['relation: ' + cause + '_' + effect]['llm_label'] = row['llm_class']


      graph.nodes['event:' + cause]['embedding'] = np.asarray(row['Embedding_' + cause_col], dtype=np.float64)
      graph.nodes['event:' + effect]['embedding'] = np.asarray(row['Embedding_' + effect_col], dtype=np.float64)
      graph.nodes['relation: ' + cause + '_' + effect]['embedding'] = np.mean([np.asarray(row['Embedding_' + cause_col], dtype=np.float64),np.asarray(row['Embedding_' + effect_col], dtype=np.float64)], axis=0)

   return graph
              
def generate_heterogeneous_graph(df, model, dic_who, dic_when, dic_where, cause_col = ' Cause', effect_col = ' Effect'):

   graph = nx.Graph()

   df[cause_col] = df[cause_col].astype(str)
   df[effect_col] = df[effect_col].astype(str)

   df['Embedding_' + cause_col] = list(model.encode(df[cause_col]))
   df['Embedding_' + effect_col] = list(model.encode(df[effect_col]))

   for _, row in df.iterrows(): 
      
      cause = row[cause_col]
      effect = row[effect_col]
      
      graph.add_edge('event:' + cause, 'relation: ' + cause + '_' + effect)
      graph.add_edge('relation: ' + cause + '_' + effect, 'event:' + effect)
      
      graph.nodes['event:' + cause]['label'] = 0#'aux'
      graph.nodes['event:' + effect]['label'] = 0#'aux'
      graph.nodes['relation: ' + cause + '_' + effect]['label'] = 1#row['Label']

      graph.nodes['event:' + cause]['embedding'] = np.asarray(row['Embedding_' + cause_col], dtype=np.float64)
      graph.nodes['event:' + effect]['embedding'] = np.asarray(row['Embedding_' + effect_col], dtype=np.float64)
      graph.nodes['relation: ' + cause + '_' + effect]['embedding'] = np.mean([np.asarray(row['Embedding_' + cause_col], dtype=np.float64),np.asarray(row['Embedding_' + effect_col], dtype=np.float64)], axis=0)
      
      topic_cause = row['Topic_Cause']
      topic_effect = row['Topic_Effect']
      try:
         who_cause = eval(row['5w1h_cause'])['Who']
         where_cause = eval(row['5w1h_cause'])['Where']
         when_cause = eval(row['5w1h_cause'])['When']
      except:
         who_cause = ''
         where_cause = ''
         when_cause = ''
      try:
         who_effect = eval(row['5w1h_effect'])['Who']
         where_effect = eval(row['5w1h_effect'])['Where']
         when_effect = eval(row['5w1h_effect'])['When']
      except:
         who_effect = ''
         where_effect = ''
         when_effect = ''

      if topic_cause != '-1_the_in_and_of': 
         graph.add_edge('event:' + cause, 'topic:' + topic_cause)
         graph.add_edge('topic:' + topic_cause, 'event:' + cause)
         graph.nodes['topic:' + topic_cause]['embedding'] = np.asarray(model.encode(topic_cause), dtype=np.float64)
         graph.nodes['topic:' + topic_cause]['label'] = 2#'topic'

      if topic_effect != '-1_the_in_and_of': 
         graph.add_edge('event:' + effect, 'topic:' + topic_effect)
         graph.add_edge('topic:' + topic_effect,'event:' + effect)
         graph.nodes['topic:' + topic_effect]['embedding'] = np.asarray(model.encode(topic_effect), dtype=np.float64)
         graph.nodes['topic:' + topic_effect]['label'] = 2#'topic'
         
      try:
         graph.add_edge('event:' + cause, 'who:' + dic_who[who_cause])
         graph.add_edge('who:' + dic_who[who_cause],'event:' + cause)
         graph.nodes['who:' + dic_who[who_cause]]['embedding'] = np.asarray(model.encode(dic_who[who_cause]), dtype=np.float64)
         graph.nodes['who:' + dic_who[who_cause]]['label'] = 3#'who'
      except: 
         print('Dict who without the key: ' + who_cause)

      try:
         graph.add_edge('event:' + cause, 'where:' + dic_where[where_cause])
         graph.add_edge('where:' + dic_where[where_cause], 'event:' + cause)
         graph.nodes['where:' + dic_where[where_cause]]['embedding'] = np.asarray(model.encode(dic_where[where_cause]), dtype=np.float64)
         graph.nodes['where:' + dic_where[where_cause]]['label'] = 4#'where'
      except: 
         print('Dict where without the key: ' + where_cause)

      try:
         wcs = dic_when[when_cause]
         for wc in wcs:
            graph.add_edge('event:' + cause, 'when:' + wc)
            graph.add_edge('when:' + wc, 'event:' + cause)
            graph.nodes['when:' + wc]['embedding'] = np.asarray(model.encode(wc), dtype=np.float64)
            graph.nodes['when:' + wc]['label'] = 5#'when'
      except: 
         print('Dict wheen without the key: ' + when_cause)

      try:
         graph.add_edge('event:' + effect, 'who:' + dic_who[who_effect])
         graph.add_edge('who:' + dic_who[who_effect],'event:' + effect)
         graph.nodes['who:' + dic_who[who_effect]]['embedding'] = np.asarray(model.encode(dic_who[who_effect]), dtype=np.float64)
         graph.nodes['who:' + dic_who[who_effect]]['label'] = 3#'who'
      except: 
         print('Dict who without the key: ' + who_effect)

      try:
         graph.add_edge('event:' + effect, 'where:' + dic_where[where_effect])
         graph.add_edge('where:' + dic_where[where_effect],'event:' + effect)
         graph.nodes['where:' + dic_where[where_effect]]['embedding'] = np.asarray(model.encode(dic_where[where_effect]), dtype=np.float64)
         graph.nodes['where:' + dic_where[where_effect]]['label'] = 4#'where'
      except: 
         print('Dict where without the key: ' + where_effect)

      try:
         wes = dic_when[when_effect]
         for we in wes:
            graph.add_edge('event:' + effect, 'when:' + we)
            graph.add_edge('when:' + we,'event:' + effect)
            graph.nodes['when:' + we]['embedding'] = np.asarray(model.encode(we), dtype=np.float64)
            graph.nodes['when:' + we]['label'] = 5#'when'
      except: 
         print('Dict wheen without the key: ' + when_effect)
   
   return graph

from plotly import graph_objs as go

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
    node_labels.append(G.nodes[node]['label'])

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
# One-Class Edge Classification through Heterogeneous Hypergraph for Causal Discovery

- Marcos Gôlo (ICMC/USP) | marcosgolo@usp.br
- Ricardo Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Citing:

Under Review in Scientific Reports Journal

# Abstract 

We explore the problem of causal discovery between event pairs. Existing LLM-based methods analyze event relationships in isolation, with a gap in exploring relationships between multiple events collectively. Graph neural networks (GNNs) address this gap by modeling event relationships as graph edges. However, they also introduce another gap, as they are inherently designed to learn representations for nodes rather than edges. Additionally, the binary GNN classification requires labeling a large number of causal and non-causal examples, and the scope of non-causal relationships is vast. To address these challenges, we propose eCHOLGA (edge Classification through Heterogeneous One-cLass Graph Autoencoder)). Our method leverages heterogeneous hypergraphs—transforming edges into nodes—to better capture edge (causal relationship) representations using GNNs, and incorporates additional node and edge types, which helps mitigate issues related to disconnected hypergraphs. Furthermore, our approach employs a one-class learning strategy, where we learn only from causal examples, thereby reducing the labeling burden. Beyond its effectiveness, eCHOLGA is also interpretable, improving the understanding of the causal discovery learning process. Experimental results demonstrate that eCHOLGA was competitive and outperformed state-of-the-art methods, establishing it as a promising approach for causal discovery in event pairs.

# emergent_topology_of_ideas_in_vision
A repo for persistent homology analysis of ideas in vision transformers, using the attention mechanism to derive the Jensen-Shannon distance between probability distributions associated to tokens. 

## How it Works
1. Obtain the attention matrix from model using the input tokens $X \in \mathbb{R}^{d \times n}$.
2. Compute softmax of attention matrix, $\text{Attn}(X)$, to get probability distributions $P(X_i)$ associated to each token embedding $X_i$. 
3. Compute pairwise Jensen-Shannon divergence (should probably change to distance metric) between the probability distributions associated to tokens. 
4. Use the distance matrix to compute persistent homology
5. For each value of the scale-parameter use for the persistent homology computation, plot the $1$-skeleton of the simplicial complex obtained for that scale-parameter value. 

Persistent homology analysis can be useful in studying and improving vision transformers and attention mechanisms in several ways:

1. **Topological feature analysis**: Persistent homology helps quantify the topological features of data, such as connected components, loops, and voids. By analyzing these features in the context of attention mechanisms in vision transformers, we can better understand how the model captures and processes the structure and patterns within images.

2. **Understanding attention behavior**: Persistent homology can provide insights into the distribution and stability of attention weights in different layers and heads of a vision transformer. This can help researchers identify whether the attention mechanism effectively focuses on important regions or patterns in an image, and whether it generalizes well across different inputs.

3. **Model robustness and interpretability**: By comparing the persistent homology analysis of images before and after applying adversarial perturbations or other transformations, we can evaluate the robustness of vision transformers to such perturbations. Furthermore, understanding how topological features are captured by the attention mechanism can improve the interpretability of the model and help explain its decisions.

# Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare

Murphy's Probabilistic Machine Learning claims one framework covers every graph embedding method: an encoder, a decoder, and a loss with three weights (GraphEDM). This project tests that, in tinygrad, on the methods the site's other GNN projects leave out, using graphs where the truth is known: a degree-corrected contextual stochastic block model that keeps its true edge probabilities, and a balanced tree with an exact metric. You write one trainer that is literally the GraphEDM loss, and every neural method is a call to it. Label propagation by iteration and in closed form, where one class swallows the graph at 1% labels until class mass normalisation repairs it, then label spreading. Laplacian eigenmaps, where one tiny component costs an embedding dimension. DeepWalk's similarity matrix, with its window weights derived and checked by simulation because the printed expression is not a distribution, then factorised as NetMF with no walk ever sampled. GAE and VGAE, including why an inner-product decoder cannot learn without reweighting positives, and what the KL weight really does. Deep Graph Infomax, judged by a linear probe against its own untrained encoder and against plain feature smoothing. A GCN classifier with an optional reconstruction term, and Planetoid. Poincare embeddings with Riemannian SGD, against Euclidean embeddings and classical MDS at equal dimension. Headline numbers, means over ten seeds on the prod CPU image with 1000 nodes and five classes. Accuracy at 2 and 10 labels per class: raw-feature probe 0.35 and 0.51, harmonic label propagation 0.28 and 0.69, with class mass normalisation 0.75 and 0.83, label spreading 0.61 and 0.79, Planetoid 0.59 and 0.68, GAE probe 0.69 and 0.84, untrained two-hop smoothing 0.82 and 0.92, DGI probe 0.85 and 0.92, GCN 0.83 and 0.93. Adding reconstruction to the GCN lowered accuracy on all ten seeds (0.83 to 0.66 at 2 labels, beta = 1). Joint Planetoid rarely beat its own two-stage pipeline (2 of 10 seeds at 2 labels, none at 10), so the book's motivation for it did not reproduce here. Held-out link AUC: the true probabilities 0.86, common neighbours 0.55, NetMF 0.69, GAE 0.72, VGAE 0.71 at the ELBO's own KL scale, 0.77 with the KL turned up 100 times, and 0.60 at 1000 times as the posterior nearly collapses. On a heterophilous graph label propagation falls below chance at 0.16, a GCN reaches 0.30, and a linear model on raw features 0.54. On a 364-node tree, mean average precision is 0.94 for 2-D Poincare, 0.68 for 2-D Euclidean, 0.79 for 10-D Euclidean and 0.04 for 2-D MDS, though Euclidean has the better mean rank from 5 dimensions up. All data is synthetic, with one Gaussian mean per class, so read these as results about this family of graphs. The assembled script runs the whole study for one seed in three to five minutes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** contextual_sbm
- [x] **2.** edge_homophily
- [x] **3.** balanced_tree
- [x] **4.** label_propagation
- [x] **5.** harmonic_labels
- [x] **6.** class_mass_normalize
- [x] **7.** label_spreading
- [x] **8.** kmeans
- [x] **9.** laplacian_eigenmaps
- [x] **10.** deepwalk_similarity
- [x] **11.** netmf_embedding
- [x] **12.** edm_fit
- [x] **13.** GCNEncoder
- [x] **14.** link_split
- [x] **15.** link_auc
- [x] **16.** gae_full_loss
- [x] **17.** train_gae
- [x] **18.** train_vgae

---

Built on Deep-ML.

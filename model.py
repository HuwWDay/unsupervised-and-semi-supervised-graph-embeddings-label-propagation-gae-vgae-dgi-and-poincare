"""
Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - contextual_sbm
import numpy as np


def contextual_sbm(
    n=1000,
    k=5,
    avg_degree=5.0,
    homophily=0.8,
    mu=2.0,
    d=64,
    degree_shape=2.5,
    seed=0,
):
    rng = np.random.default_rng([seed, 23])

    # 1. Labels
    y = rng.permutation(np.arange(n) % k).astype(np.int64)

    # 2. Degree propensities
    if degree_shape > 0:
        theta = rng.pareto(degree_shape, n) + 1.0
        theta = np.minimum(theta, np.sqrt(n) / 2)
        theta = theta / np.mean(theta)
    else:
        theta = np.ones(n, dtype=np.float64)

    # 3. Block rates & edge probability matrix P
    p_in = avg_degree * homophily / (n / k)
    p_out = avg_degree * (1.0 - homophily) / (n - n / k)

    same_label = y[:, None] == y[None, :]
    block_rates = np.where(same_label, p_in, p_out)

    P = block_rates * np.outer(theta, theta)
    P = np.clip(P, 0.0, 1.0)
    np.fill_diagonal(P, 0.0)

    # 4. Edges
    triu_mask = np.triu(rng.random((n, n)) < P, 1)
    A = (triu_mask | triu_mask.T).astype(np.float64)

    # 5. Class means
    M = rng.normal(size=(k, d))
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

    # 6. Features
    X = (mu * M[y] + rng.normal(size=(n, d))).astype(np.float32)

    return {"A": A, "X": X, "y": y, "P": P}

# Step 2 - edge_homophily
import numpy as np


def edge_homophily(A, y):
    i, j = np.nonzero(np.triu(A > 0, 1))
    if len(i) == 0:
        return 0.0
    return float(np.mean(y[i] == y[j]))


def few_labels(y, n_per_class, seed=0):
    rng = np.random.default_rng([seed, 29])
    num_classes = int(np.max(y)) + 1

    selected_per_class = []
    for c in range(num_classes):
        class_indices = np.flatnonzero(y == c)
        chosen = rng.choice(class_indices, n_per_class, replace=False)
        selected_per_class.append(chosen)

    labeled = np.concatenate(selected_per_class)

    mask = np.ones(len(y), dtype=bool)
    mask[labeled] = False
    rest = np.flatnonzero(mask)

    return labeled, rest

# Step 3 - balanced_tree
import numpy as np


def balanced_tree(b, depth):
    if b == 1:
        N = depth + 1
    else:
        N = (b ** (depth + 1) - 1) // (b - 1)

    parent = np.empty(N, dtype=np.int64)
    level = np.empty(N, dtype=np.int64)
    parent[0] = -1
    level[0] = 0

    if N > 1:
        # Under 0-indexed BFS ordering, children of node p are p * b + 1 to p * b + b
        # Inverting this gives: parent(i) = (i - 1) // b
        indices = np.arange(1, N, dtype=np.int64)
        parent[1:] = (indices - 1) // b
        for i in range(1, N):
            level[i] = level[parent[i]] + 1

    A = np.zeros((N, N), dtype=np.float64)
    if N > 1:
        nodes = np.arange(1, N, dtype=np.int64)
        pars = parent[1:]
        A[nodes, pars] = 1.0
        A[pars, nodes] = 1.0

    return A, parent, level


def tree_distances(parent):
    N = len(parent)
    D = np.zeros((N, N), dtype=np.int64)

    # For u < v: d(v, u) = d(parent[v], u) + 1
    for v in range(1, N):
        p = parent[v]
        D[v, :v] = D[p, :v] + 1
        D[:v, v] = D[v, :v]

    return D

# Step 4 - label_propagation
import numpy as np


def label_propagation(W, labeled, y_labeled, k, n_iter=100):
    N = W.shape[0]

    # Initialize F as an (N, k) float64 matrix with one-hot rows at labeled nodes
    F = np.zeros((N, k), dtype=np.float64)
    F[labeled, y_labeled] = 1.0

    # Row-normalize W to obtain the transition matrix P
    row_sums = W.sum(axis=1, keepdims=True)
    P = np.divide(W, row_sums, out=np.zeros_like(W, dtype=np.float64), where=(row_sums != 0))

    # Keep a copy of the fixed labeled one-hot rows
    F_labeled = F[labeled].copy()

    # Iterate propagation and reset clamped nodes
    for _ in range(n_iter):
        F = P @ F
        F[labeled] = F_labeled

    return F


def dirichlet_energy(W, F):
    # D is the degree matrix: diag(W @ 1)
    degrees = W.sum(axis=1)
    # L = D - W; (D - W) @ F can be written as degrees[:, None] * F - W @ F
    LF = degrees[:, None] * F - (W @ F)
    # tr(F.T @ LF)
    energy = 2.0 * np.sum(F * LF)
    return float(energy)

# Step 5 - harmonic_labels
import numpy as np


def harmonic_labels(W, labeled, y_labeled, k):
    N = W.shape[0]
    F = np.zeros((N, k), dtype=np.float64)

    # Set the labeled rows to their one-hot vectors
    F[labeled, y_labeled] = 1.0

    # Determine unlabeled indices in increasing order
    is_labeled = np.zeros(N, dtype=bool)
    is_labeled[labeled] = True
    unlabeled = np.flatnonzero(~is_labeled)

    if len(unlabeled) == 0:
        return F

    # Full graph degrees D (row sums of full W)
    degrees = W.sum(axis=1)

    # Extract submatrices for unlabeled (U) and labeled (L) blocks
    W_UU = W[np.ix_(unlabeled, unlabeled)]
    W_UL = W[np.ix_(unlabeled, labeled)]
    D_UU = np.diag(degrees[unlabeled])

    # Construct linear system: (D_UU - W_UU + 1e-9 * I) F_U = W_UL Y_L
    A = D_UU - W_UU + 1e-9 * np.eye(len(unlabeled), dtype=np.float64)
    B = W_UL @ F[labeled]

    F[unlabeled] = np.linalg.solve(A, B)

    return F

# Step 6 - class_mass_normalize
import numpy as np


def class_mass_normalize(F, y_labeled, k):
    counts = np.bincount(y_labeled, minlength=k)[:k]
    q = (counts + 1.0) / (len(y_labeled) + k)
    m = np.maximum(F.sum(axis=0), 1e-12)
    return F * (q / m)


def node_accuracy(F, y, idx):
    if len(idx) == 0:
        return 0.0
    preds = F.argmax(axis=1)
    return float(np.mean(preds[idx] == y[idx]))

# Step 7 - label_spreading
import numpy as np


def _normalized_adjacency(W):
    W = np.asarray(W, dtype=np.float64)
    deg = W.sum(axis=1)
    inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return W * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]


def label_spreading(W, labeled, y_labeled, k, alpha=0.9):
    N = W.shape[0]
    S = _normalized_adjacency(W)

    Y = np.zeros((N, k), dtype=np.float64)
    Y[labeled, y_labeled] = 1.0

    A = np.eye(N, dtype=np.float64) - alpha * S
    F = (1.0 - alpha) * np.linalg.solve(A, Y)
    return F


def spreading_objective(W, F, Y, mu):
    F = np.asarray(F, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    S = _normalized_adjacency(W)

    smoothness = np.sum(F * (F - S @ F))
    fitting = mu * np.sum((F - Y) ** 2)
    return float(smoothness + fitting)


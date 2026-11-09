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


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

# Step 8 - kmeans
import numpy as np


def kmeans(Z, k, seed=0, n_init=5, n_iter=50):
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    rng = np.random.default_rng(seed)

    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        # 1. k-means++ seeding
        centers = np.empty((k, d), dtype=np.float64)
        centers[0] = Z[rng.integers(n)]

        d2 = np.sum((Z - centers[0]) ** 2, axis=1)
        for c in range(1, k):
            total_d2 = d2.sum()
            if total_d2 == 0.0:
                idx = rng.integers(n)
            else:
                idx = rng.choice(n, p=d2 / total_d2)
            centers[c] = Z[idx]
            new_d2 = np.sum((Z - centers[c]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)

        # 2. Lloyd iterations
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(n_iter):
            # Pairwise squared Euclidean distances: (n, k)
            dists = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int64)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centers[c] = Z[mask].mean(axis=0)

        # Compute within-cluster sum of squares (inertia)
        inertia = float(np.sum((Z - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def adjusted_rand_index(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = len(a)

    if n < 2:
        return 1.0

    # Build contingency table n_ij
    max_a = int(a.max()) + 1
    max_b = int(b.max()) + 1
    contingency = np.zeros((max_a, max_b), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)

    def comb2(x):
        return x * (x - 1) / 2.0

    sum_comb_nij = np.sum(comb2(contingency))
    a_sums = contingency.sum(axis=1)
    b_sums = contingency.sum(axis=0)

    sum_comb_ai = np.sum(comb2(a_sums))
    sum_comb_bj = np.sum(comb2(b_sums))
    comb_n = comb2(n)

    expected_index = (sum_comb_ai * sum_comb_bj) / comb_n
    max_index = 0.5 * (sum_comb_ai + sum_comb_bj)
    denom = max_index - expected_index

    if denom == 0.0:
        return 1.0 if sum_comb_nij == expected_index else 0.0

    return float((sum_comb_nij - expected_index) / denom)

# Step 9 - laplacian_eigenmaps
import numpy as np


def laplacian_eigenmaps(W, dim, tau=0.0):
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    # Regularized adjacency matrix Wt = W + tau / N
    Wt = W + (tau / N)

    # Node degrees D (row sums of Wt) and D^{-1/2} with zero-degree guard
    deg = Wt.sum(axis=1)
    inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    # Normalized Laplacian: L_sym = I - D^{-1/2} Wt D^{-1/2}
    S = Wt * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]
    L_sym = np.eye(N, dtype=np.float64) - S

    # Eigendecomposition in ascending order
    lam, V = np.linalg.eigh(L_sym)

    # Map eigenvectors back via Z = D^{-1/2} V, skipping the first trivial eigenvector
    Z = inv_sqrt_deg[:, None] * V[:, 1 : 1 + dim]

    return Z, lam

# Step 10 - deepwalk_similarity
import numpy as np


def window_weights(T, window="dynamic"):
    if window == "fixed":
        return np.full(T, 1.0 / T, dtype=np.float64)
    elif window == "dynamic":
        # Under uniform window size w ~ Uniform{1, ..., T}, a context at distance q
        # is included whenever w >= q (which happens for T - q + 1 choices of w).
        # Normalizing by sum_{q=1}^T (T - q + 1) = T(T + 1) / 2 gives:
        q = np.arange(1, T + 1, dtype=np.float64)
        weights = T - q + 1.0
        return weights / np.sum(weights)
    else:
        raise ValueError(f"Unknown window type: {window!r}")


def deepwalk_similarity(W, T, window="dynamic"):
    W = np.asarray(W, dtype=np.float64)
    N = W.shape[0]

    # Row-normalize W to get transition matrix P = D^{-1} W
    row_sums = W.sum(axis=1, keepdims=True)
    P = np.divide(W, row_sums, out=np.zeros_like(W, dtype=np.float64), where=(row_sums != 0))

    weights = window_weights(T, window=window)

    S = np.zeros((N, N), dtype=np.float64)
    P_power = P.copy()
    for q in range(T):
        S += weights[q] * P_power
        if q < T - 1:
            P_power = P_power @ P

    return S

# Step 11 - netmf_embedding
import numpy as np


def netmf_embedding(W, T=5, b=1.0, dim=16):
    W = np.asarray(W, dtype=np.float64)

    # 1. Compute DeepWalk similarity and total volume
    S = deepwalk_similarity(W, T)
    vol = np.sum(W)

    # 2. Form M = (vol / b) * S * D^{-1} and L = log(max(M, 1))
    deg = W.sum(axis=1)
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
    M = (vol / b) * (S * inv_deg[None, :])
    L = np.log(np.maximum(M, 1.0))

    # 3. Symmetrise L and compute its eigendecomposition
    L = 0.5 * (L + L.T)
    eigvals, eigvecs = np.linalg.eigh(L)

    # 4. Keep top `dim` eigenpairs sorted by decreasing |lambda|
    top_idx = np.argsort(np.abs(eigvals))[::-1][:dim]
    lam = eigvals[top_idx]
    V = eigvecs[:, top_idx]

    Zs = V * np.sqrt(np.abs(lam))[None, :]
    Zt = Zs * np.sign(lam)[None, :]

    return Zs, Zt

# Step 12 - edm_fit
import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn.optim import Adam


def edm_fit(
    params,
    sup=None,
    recon=None,
    reg=None,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
    batch_fn=None,
    steps=200,
    lr=0.01,
    seed=0,
):
    use_sup = sup is not None and alpha != 0
    use_recon = recon is not None and beta != 0
    use_reg = reg is not None and gamma != 0

    if not (use_sup or use_recon or use_reg):
        raise ValueError("At least one loss term must be active (callable not None and weight != 0).")

    rng = np.random.default_rng(seed)

    for p in params:
        p.requires_grad = True

    opt = Adam(params, lr=lr, b1=0.9, b2=0.999, eps=1e-8)

    @TinyJit
    def train_step(*batch_tensors):
        opt.zero_grad()

        loss = None
        if use_sup:
            term = alpha * sup()
            loss = term if loss is None else loss + term
        if use_recon:
            term = beta * recon(*batch_tensors)
            loss = term if loss is None else loss + term
        if use_reg:
            term = gamma * reg()
            loss = term if loss is None else loss + term

        loss.backward()
        opt.step()
        return loss.realize()

    record_steps = {0, steps - 1}
    for t in range(steps):
        if (t + 1) % 50 == 0:
            record_steps.add(t)

    history = []
    with Tensor.train():
        for t in range(steps):
            if use_recon and batch_fn is not None:
                batch_np = batch_fn(rng)
                batch_tensors = tuple(Tensor(arr) for arr in batch_np)
            else:
                batch_tensors = ()

            loss_tensor = train_step(*batch_tensors)

            if t in record_steps:
                history.append((int(t), float(loss_tensor.item())))

    return history

# Step 13 - GCNEncoder
import math
import numpy as np
from tinygrad import Tensor


def gcn_normalize(A):
    A = np.asarray(A, dtype=np.float32)
    N = A.shape[0]
    A_tilde = A + np.eye(N, dtype=np.float32)
    deg = A_tilde.sum(axis=1)
    inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    Ahat = A_tilde * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]
    return Ahat.astype(np.float32)


def glorot(rng, a, b):
    limit = math.sqrt(6.0 / (a + b))
    weights = (rng.uniform(-1.0, 1.0, (a, b)) * limit).astype(np.float32)
    return Tensor(weights).contiguous().realize()


class GCNEncoder:
    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = glorot(rng, in_dim, hidden)
        self.W2 = glorot(rng, hidden, out_dim)

    def parameters(self):
        return [self.W1, self.W2]

    def hidden_layer(self, Ahat, X):
        return (Ahat @ X @ self.W1).relu()

    def __call__(self, Ahat, X):
        H = self.hidden_layer(Ahat, X)
        return Ahat @ H @ self.W2

# Step 14 - link_split
import numpy as np


def link_split(A, val_frac=0.05, test_frac=0.10, seed=0):
    rng = np.random.default_rng([seed, 31])
    N = A.shape[0]

    # 1. List undirected edges (i < j) and shuffle
    i, j = np.nonzero(np.triu(A, 1))
    edges = np.column_stack((i, j)).astype(np.int64)
    edges = edges[rng.permutation(len(edges))]

    # 2. Slice into validation, test, and training positive edges
    E_count = len(edges)
    nv = int(E_count * val_frac)
    nt = int(E_count * test_frac)

    val_pos = edges[:nv]
    test_pos = edges[nv : nv + nt]
    train = edges[nv + nt :]

    # Build symmetric A_train holding only training edges
    A_train = np.zeros((N, N), dtype=np.float64)
    if len(train) > 0:
        A_train[train[:, 0], train[:, 1]] = 1.0
        A_train[train[:, 1], train[:, 0]] = 1.0

    # 3. Sample non-overlapping negative edges for validation, then test
    seen_neg = set()

    def sample_negatives(count):
        negs = []
        while len(negs) < count:
            a, b = rng.integers(0, N, 2)
            if a == b:
                continue
            u, v = (int(a), int(b)) if a < b else (int(b), int(a))
            if A[u, v] == 0 and (u, v) not in seen_neg:
                seen_neg.add((u, v))
                negs.append((u, v))
        if count == 0:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(negs, dtype=np.int64)

    val_neg = sample_negatives(nv)
    test_neg = sample_negatives(nt)

    return {
        "A_train": A_train,
        "train": train,
        "val_pos": val_pos,
        "test_pos": test_pos,
        "val_neg": val_neg,
        "test_neg": test_neg,
    }

# Step 15 - link_auc
import numpy as np


def link_auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()

    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    scores = np.concatenate([pos, neg])
    n = len(scores)

    # Sort all scores to assign 1-based average ranks in O(n log n)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    # Identify starts and ends of tied groups
    diff = np.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = sorted_scores[1:] != sorted_scores[:-1]
    starts = np.flatnonzero(diff)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1

    # Average 1-based rank for each tied block: (start + 1 + end + 1) / 2
    avg_ranks_per_block = 0.5 * (starts + ends + 2.0)
    block_lengths = ends - starts + 1
    sorted_ranks = np.repeat(avg_ranks_per_block, block_lengths)

    # Map ranks back to original positions
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks

    # Mann-Whitney U statistic for the positive class
    u_pos = np.sum(ranks[:n_pos]) - (n_pos * (n_pos + 1.0)) / 2.0
    return float(u_pos / (n_pos * n_neg))


def pair_auc(S, pos_pairs, neg_pairs):
    pos_scores = S[pos_pairs[:, 0], pos_pairs[:, 1]]
    neg_scores = S[neg_pairs[:, 0], neg_pairs[:, 1]]
    return link_auc(pos_scores, neg_scores)

# Step 16 - gae_full_loss
import numpy as np
from tinygrad import Tensor


def gae_full_loss(Z, A, pos_weight=None):
    N = A.shape[0]
    if pos_weight is None:
        pos_sum = A.sum()
        w = (N * N - pos_sum) / pos_sum
    else:
        w = pos_weight

    logits = Z @ Z.T
    # -[w * A * log(sigma(logits)) + (1 - A) * log(sigma(-logits))]
    loss_matrix = -(w * A * logits.logsigmoid() + (1.0 - A) * (-logits).logsigmoid())
    return loss_matrix.mean()


def gae_sampled_loss(Z, pos, neg):
    pos_logits = (Z[pos[:, 0]] * Z[pos[:, 1]]).sum(axis=1)
    neg_logits = (Z[neg[:, 0]] * Z[neg[:, 1]]).sum(axis=1)

    pos_loss = (-pos_logits.logsigmoid()).mean()
    neg_loss = (-(-neg_logits).logsigmoid()).mean()
    return pos_loss + neg_loss

# Step 17 - train_gae
import numpy as np
from tinygrad import Tensor


def train_gae(split, X, hidden=32, dim=16, loss="sampled", steps=200, lr=0.01, seed=0):
    A_train = split["A_train"]
    N = A_train.shape[0]

    # 1. Precompute normalized adjacency and node feature tensors
    Ahat = Tensor(gcn_normalize(A_train)).realize()
    Xt = Tensor(np.asarray(X, dtype=np.float32)).realize()

    # 2. Initialize GCN encoder
    enc = GCNEncoder(X.shape[1], hidden, dim, seed)

    # 3. Prepare positive training edges tensor
    pos = Tensor(np.asarray(split["train"], dtype=np.int32)).realize()
    m = split["train"].shape[0]

    # 4. Configure reconstruction loss and optional batch sampling
    if loss == "sampled":
        def batch_fn(rng):
            return (rng.integers(0, N, (m, 2), dtype=np.int32),)

        def recon(neg):
            return gae_sampled_loss(enc(Ahat, Xt), pos, neg)

    elif loss == "full":
        At = Tensor(np.asarray(A_train, dtype=np.float32)).realize()
        batch_fn = None

        def recon():
            return gae_full_loss(enc(Ahat, Xt), At)

    elif loss == "full_unweighted":
        At = Tensor(np.asarray(A_train, dtype=np.float32)).realize()
        batch_fn = None

        def recon():
            return gae_full_loss(enc(Ahat, Xt), At, pos_weight=1.0)

    else:
        raise ValueError(f"Unknown loss mode: {loss!r}")

    hist = edm_fit(
        enc.parameters(),
        recon=recon,
        alpha=0.0,
        beta=1.0,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed,
    )

    Z = enc(Ahat, Xt).numpy()

    return {"Z": Z, "enc": enc, "hist": hist}

# Step 18 - train_vgae
import numpy as np
from tinygrad import Tensor


def gaussian_kl(mu, logvar):
    # KL(N(mu, diag(exp(logvar))) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # Summed over latent dimensions (axis=1), averaged over nodes (axis=0)
    kl_per_node = -0.5 * (1.0 + logvar - mu * mu - logvar.exp()).sum(axis=1)
    return kl_per_node.mean()


def train_vgae(split, X, hidden=32, dim=16, kl_weight=1.0, steps=200, lr=0.01, seed=0):
    A_train = split["A_train"]
    N = A_train.shape[0]

    Ahat = Tensor(gcn_normalize(A_train)).realize()
    Xt = Tensor(np.asarray(X, dtype=np.float32)).realize()

    enc = GCNEncoder(X.shape[1], hidden, dim, seed)
    W3 = glorot(np.random.default_rng(seed + 1000), hidden, dim)

    pos = Tensor(np.asarray(split["train"], dtype=np.int32)).realize()
    m = split["train"].shape[0]

    def encode():
        H = Ahat @ enc.hidden_layer(Ahat, Xt)
        mu = H @ enc.W2
        logvar = (H @ W3).clip(-10.0, 10.0)
        return mu, logvar

    def batch_fn(rng):
        neg = rng.integers(0, N, (m, 2), dtype=np.int32)
        eps = rng.normal(0.0, 1.0, (N, dim)).astype(np.float32)
        return (neg, eps)

    def recon(neg, eps):
        mu, logvar = encode()
        z = mu + (0.5 * logvar).exp() * eps
        return gae_sampled_loss(z, pos, neg)

    def reg():
        mu, logvar = encode()
        return gaussian_kl(mu, logvar) / N

    hist = edm_fit(
        [enc.W1, enc.W2, W3],
        recon=recon,
        reg=reg,
        alpha=0.0,
        beta=1.0,
        gamma=kl_weight,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed,
    )

    mu_final, logvar_final = encode()
    mu_np = mu_final.numpy()
    logvar_np = logvar_final.numpy()
    kl_per_dim = (-0.5 * (1.0 + logvar_np - mu_np ** 2 - np.exp(logvar_np))).mean(axis=0)

    return {
        "mu": mu_np,
        "logvar": logvar_np,
        "kl_per_dim": kl_per_dim,
        "hist": hist,
    }

# Step 19 - dgi_loss
import numpy as np
from tinygrad import Tensor


def readout(Z):
    return Z.mean(axis=0).sigmoid()


def dgi_loss(Z, Z_neg, Wb, fake_summary=False):
    s = readout(Z)
    s_neg = readout(Z_neg) if fake_summary else s

    # Bilinear logits: z_i^T Wb s = (Z @ (Wb @ s))
    pos_logits = Z @ (Wb @ s)
    neg_logits = Z_neg @ (Wb @ s_neg)

    # -0.5 * [ mean(log(sigma(pos_logits))) + mean(log(1 - sigma(neg_logits))) ]
    # Note: log(1 - sigma(x)) = log(sigma(-x)) = (-x).logsigmoid()
    pos_loss = pos_logits.logsigmoid().mean()
    neg_loss = (-neg_logits).logsigmoid().mean()

    return -0.5 * (pos_loss + neg_loss)

# Step 20 - train_dgi
import numpy as np
from tinygrad import Tensor


def train_dgi(A, X, dim=256, steps=200, lr=0.001, seed=0):
    N = A.shape[0]

    # 1. Precompute normalized adjacency and node feature tensors
    Ahat = Tensor(gcn_normalize(A)).realize()
    Xt = Tensor(np.asarray(X, dtype=np.float32)).realize()

    # 2. Initialize GCN encoder and bilinear discriminator weights Wb
    enc = GCNEncoder(X.shape[1], dim, dim, seed)
    Wb = glorot(np.random.default_rng(seed + 1000), dim, dim)

    def embed(x):
        return enc(Ahat, x).relu()

    # 3. Train via edm_fit with alpha=0.0, beta=1.0 (if steps > 0)
    if steps > 0:
        def batch_fn(rng):
            return (rng.permutation(N).astype(np.int32),)

        def recon(perm):
            return dgi_loss(embed(Xt), embed(Xt[perm]), Wb)

        hist = edm_fit(
            [enc.W1, enc.W2, Wb],
            recon=recon,
            alpha=0.0,
            beta=1.0,
            batch_fn=batch_fn,
            steps=steps,
            lr=lr,
            seed=seed,
        )
    else:
        hist = []

    Z = embed(Xt).numpy()
    return {"Z": Z, "hist": hist}

# Step 21 - linear_probe
import numpy as np
from tinygrad import Tensor


def linear_probe(Z, y, train_idx, test_idx, l2=1e-3, steps=300, lr=0.05, seed=0):
    Z = np.asarray(Z, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # 1. Standardise each column over all nodes as float32
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True) + 1e-6
    Z_std = ((Z - mean) / std).astype(np.float32)

    L = Z_std.shape[1]
    k = int(y.max()) + 1

    # 2. Fit softmax regression on train_idx
    W = Tensor.zeros(L, k).contiguous().realize()
    b = Tensor.zeros(k).contiguous().realize()

    X_train = Tensor(Z_std[train_idx]).realize()
    y_train = Tensor(y[train_idx].astype(np.int32)).realize()

    def sup():
        logits = X_train @ W + b
        return logits.sparse_categorical_crossentropy(y_train)

    def reg():
        return (W * W).sum()

    edm_fit(
        [W, b],
        sup=sup,
        reg=reg,
        alpha=1.0,
        gamma=l2,
        steps=steps,
        lr=lr,
        seed=seed,
    )

    # 3. Evaluate accuracy on test_idx
    if len(test_idx) == 0:
        return 0.0

    X_test = Tensor(Z_std[test_idx])
    preds = (X_test @ W + b).numpy().argmax(axis=1)
    return float(np.mean(preds == y[test_idx]))


def smoothed_features(A, X, hops=2):
    out = np.asarray(X, dtype=np.float32)
    if hops == 0:
        return out.copy()

    Ahat = gcn_normalize(A)
    for _ in range(hops):
        out = Ahat @ out
    return out

# Step 22 - train_gcn_classifier
import numpy as np
from tinygrad import Tensor


def train_gcn_classifier(A, X, y, labeled, hidden=32, dim=16, beta=0.0, wd=5e-4, steps=200, lr=0.01, seed=0):
    A = np.asarray(A)
    N = A.shape[0]
    y = np.asarray(y, dtype=np.int64)
    labeled = np.asarray(labeled, dtype=np.int32)
    k = int(y.max()) + 1

    # 1. Precompute tensors, encoder, and label head
    Ahat = Tensor(gcn_normalize(A)).realize()
    Xt = Tensor(np.asarray(X, dtype=np.float32)).realize()

    enc = GCNEncoder(X.shape[1], hidden, dim, seed)
    Wc = glorot(np.random.default_rng(seed + 1000), dim, k)

    labeled_t = Tensor(labeled).realize()
    y_labeled_t = Tensor(y[labeled].astype(np.int32)).realize()

    def sup():
        logits = enc(Ahat, Xt)[labeled_t] @ Wc
        return logits.sparse_categorical_crossentropy(y_labeled_t)

    def reg():
        return (enc.W1 * enc.W1).sum() + (enc.W2 * enc.W2).sum()

    if beta != 0:
        i, j = np.nonzero(np.triu(A, 1))
        edges = np.column_stack((i, j)).astype(np.int32)
        m = len(edges)
        pos = Tensor(edges).realize()

        def batch_fn(rng):
            return (rng.integers(0, N, (m, 2), dtype=np.int32),)

        def recon(neg):
            return gae_sampled_loss(enc(Ahat, Xt), pos, neg)
    else:
        recon = None
        batch_fn = None

    # 2. Fit with edm_fit
    edm_fit(
        enc.parameters() + [Wc],
        sup=sup,
        recon=recon,
        reg=reg,
        alpha=1.0,
        beta=beta,
        gamma=wd,
        batch_fn=batch_fn,
        steps=steps,
        lr=lr,
        seed=seed,
    )

    return (enc(Ahat, Xt) @ Wc).numpy()

# Step 23 - sample_context_pairs
import numpy as np


def context_cdf(S):
    S = np.array(S, dtype=np.float64, copy=True)
    N = S.shape[0]
    row_sums = S.sum(axis=1)

    # Replace any empty row with a self-loop on its diagonal
    zero_rows = row_sums == 0
    if np.any(zero_rows):
        idx = np.flatnonzero(zero_rows)
        S[idx, idx] = 1.0
        row_sums[idx] = 1.0

    probs = S / row_sums[:, None]
    return np.cumsum(probs, axis=1)


def sample_context_pairs(cdf, m, rng, neg_frac=0.5):
    N = cdf.shape[0]

    # 1-4: Draw random variables in the exact specified order
    i = rng.integers(0, N, m)
    u = rng.random(m)
    coin = rng.random(m)
    other = rng.integers(0, N, m)

    # Positive contexts via inverse CDF, clipped to [0, N - 1]
    j = np.clip((cdf[i] < u[:, None]).sum(axis=1), 0, N - 1)

    # Negative sampling mask
    is_neg = coin < neg_frac
    j = np.where(is_neg, other, j)
    gamma = np.where(is_neg, -1.0, 1.0)

    return i.astype(np.int32), j.astype(np.int32), gamma.astype(np.float32)

# Step 24 - train_planetoid
import numpy as np
from tinygrad import Tensor


def train_planetoid(S, X, y, labeled, zc_dim=16, hidden=8, pre=500, joint=200, wd=0.05, m=2000, lr=0.01, seed=0):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    labeled = np.asarray(labeled, dtype=np.int32)

    N, D = X.shape
    k = int(y.max()) + 1

    # Initialize Zc, W1, W2 in exact order using rng0
    rng0 = np.random.default_rng(seed)
    Zc = Tensor(rng0.normal(0.0, 0.1, (N, zc_dim)).astype(np.float32)).contiguous().realize()
    W1 = glorot(rng0, D, hidden)
    W2 = glorot(rng0, zc_dim + hidden, k)

    cdf = context_cdf(S)

    def batch_fn(rng):
        return sample_context_pairs(cdf, m, rng)

    def context_loss(i, j, gamma):
        return -((Zc[i] * Zc[j]).sum(1) * gamma).logsigmoid().mean()

    # 1. Pretraining Zc on context loss
    if pre > 0:
        edm_fit(
            [Zc],
            recon=context_loss,
            alpha=0.0,
            beta=1.0,
            batch_fn=batch_fn,
            steps=pre,
            lr=lr,
            seed=seed,
        )

    Zc_pre = Zc.numpy().copy()

    # 2. Joint training on supervised + context + L2 regularization losses
    Xt = Tensor(X).realize()
    labeled_t = Tensor(labeled).realize()
    y_labeled_t = Tensor(y[labeled].astype(np.int32)).realize()

    def forward_logits():
        h_feat = (Xt @ W1).relu()
        h = Zc.cat(h_feat, dim=1)
        return h @ W2

    def sup():
        logits_labeled = forward_logits()[labeled_t]
        return logits_labeled.sparse_categorical_crossentropy(y_labeled_t)

    def reg():
        return (W1 * W1).sum() + (W2 * W2).sum()

    if joint > 0:
        edm_fit(
            [Zc, W1, W2],
            sup=sup,
            recon=context_loss,
            reg=reg,
            alpha=1.0,
            beta=1.0,
            gamma=wd,
            batch_fn=batch_fn,
            steps=joint,
            lr=lr,
            seed=seed,
        )

    logits = forward_logits().numpy()
    Zc_final = Zc.numpy().copy()

    return {
        "logits": logits,
        "Zc": Zc_final,
        "Zc_pre": Zc_pre,
    }

# Step 25 - poincare_distance
import numpy as np
from tinygrad import Tensor


def poincare_distance(u, v, eps=1e-7):
    sq_diff = ((u - v) ** 2).sum(axis=-1)
    sq_u = (u ** 2).sum(axis=-1)
    sq_v = (v ** 2).sum(axis=-1)

    denom = (1.0 - sq_u) * (1.0 - sq_v)
    x = 1.0 + 2.0 * sq_diff / denom
    x = x.maximum(1.0 + eps)
    return (x + (x - 1.0).sqrt() * (x + 1.0).sqrt()).log()


def euclidean_distance(u, v, eps=1e-7):
    sq_diff = ((u - v) ** 2).sum(axis=-1)
    return (sq_diff + eps).sqrt()


def tree_softmax_loss(Z, i, j, neg, dist):
    z_i = Z[i]        # (m, L)
    z_j = Z[j]        # (m, L)
    z_neg = Z[neg]    # (m, K, L)

    pos_d = dist(z_i, z_j).unsqueeze(1)        # (m, 1)
    neg_d = dist(z_i.unsqueeze(1), z_neg)      # (m, K)

    logits = -pos_d.cat(neg_d, dim=1)          # (m, 1 + K), column 0 is true neighbour
    targets = Tensor.zeros(i.shape[0], dtype="int32")
    return logits.sparse_categorical_crossentropy(targets)

# Step 26 - riemannian_update
import numpy as np
from tinygrad import Tensor


def project_to_ball(Z, eps=1e-5):
    max_norm = 1.0 - eps
    norms = (Z * Z).sum(axis=-1, keepdim=True).sqrt()
    scale = (norms >= max_norm).where(max_norm / norms.maximum(1e-12), 1.0)
    return Z * scale


def riemannian_update(Z, grad, lr, eps=1e-5):
    sq_norm = (Z * Z).sum(axis=-1, keepdim=True)
    metric_inv = ((1.0 - sq_norm) ** 2) * 0.25
    Z_new = Z - lr * metric_inv * grad
    return project_to_ball(Z_new, eps=eps)

# Step 27 - train_tree_embedding
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes


def sample_tree_negatives(rng, A, u, K):
    num_nodes = len(A)
    m = len(u)
    neg = rng.integers(0, num_nodes, (m, K))

    # An entry is invalid if it is a neighbour of u or u itself
    u_col = u[:, None]
    invalid = (neg == u_col) | (A[u_col, neg] > 0)
    while np.any(invalid):
        n_bad = int(invalid.sum())
        neg[invalid] = rng.integers(0, num_nodes, n_bad)
        invalid = (neg == u_col) | (A[u_col, neg] > 0)

    return neg


def train_tree_embedding(
    A,
    dim=2,
    geometry="poincare",
    steps=2000,
    lr=10.0,
    K=20,
    burn=200,
    burn_factor=0.1,
    seed=0,
):
    A = np.asarray(A)
    N = A.shape[0]
    rng = np.random.default_rng(seed)

    # 1. Initialize Z
    Z_init = rng.uniform(-1e-3, 1e-3, (N, dim)).astype(np.float32)
    Z = Tensor(Z_init, requires_grad=True).contiguous().realize()

    # 2. Directed edges as realized int32 Tensors
    i_np, j_np = np.nonzero(A)
    i_np = i_np.astype(np.int32)
    j_np = j_np.astype(np.int32)
    m = len(i_np)

    i_t = Tensor(i_np).realize()
    j_t = Tensor(j_np).realize()

    if geometry == "poincare":
        dist = poincare_distance
    elif geometry == "euclidean":
        dist = euclidean_distance
    else:
        raise ValueError(f"Unknown geometry: {geometry!r}")

    scale = float(m) / float(N)

    # 3. TinyJit-compiled update step
    @TinyJit
    def train_step(neg_t, lr_t):
        loss = tree_softmax_loss(Z, i_t, j_t, neg_t, dist)
        grad = loss.gradient(Z)[0] * scale
        if geometry == "poincare":
            Z_next = riemannian_update(Z, grad, lr_t)
        else:
            Z_next = Z - lr_t * grad
        Z.assign(Z_next).realize()
        return loss.realize()

    record_steps = {0, steps - 1}
    for t in range(steps):
        if (t + 1) % 500 == 0:
            record_steps.add(t)

    hist = []
    # 4. Training loop
    for t in range(steps):
        neg_np = sample_tree_negatives(rng, A, i_np, K).astype(np.int32)
        neg_t = Tensor(neg_np).realize()

        curr_lr = lr * burn_factor if t < burn else lr
        lr_t = Tensor([curr_lr], dtype=dtypes.float32).realize()

        loss_t = train_step(neg_t, lr_t)

        if t in record_steps:
            hist.append((int(t), float(loss_t.item())))

    return Z.numpy(), hist

# Step 28 - reconstruction_metrics
import numpy as np


def pairwise_distances(Z, geometry):
    Z = np.asarray(Z, dtype=np.float64)
    diff = Z[:, None, :] - Z[None, :, :]
    sq_diff = np.sum(diff ** 2, axis=-1)

    if geometry == "euclidean":
        return np.sqrt(sq_diff)
    elif geometry == "poincare":
        sq_norm = np.sum(Z ** 2, axis=-1)
        denom = (1.0 - sq_norm)[:, None] * (1.0 - sq_norm)[None, :]
        x = 1.0 + 2.0 * sq_diff / denom
        x = np.maximum(x, 1.0)
        D = np.arccosh(x)
        np.fill_diagonal(D, 0.0)
        return D
    else:
        raise ValueError(f"Unknown geometry: {geometry!r}")


def reconstruction_metrics(D, A):
    D = np.asarray(D, dtype=np.float64)
    A = np.asarray(A)
    N = A.shape[0]

    all_ranks = []
    node_aps = []

    for u in range(N):
        neighbors = np.flatnonzero(A[u] > 0)
        if len(neighbors) == 0:
            continue

        # Non-neighbours are all other nodes (v != u) that are not adjacent to u
        non_neighbor_mask = A[u] == 0
        non_neighbor_mask[u] = False
        non_neighbors = np.flatnonzero(non_neighbor_mask)

        d_neigh = D[u, neighbors]
        d_non = D[u, non_neighbors]

        # Rank of v: 1 + number of non-neighbours strictly closer to u than v is
        # Compare d_non[:, None] < d_neigh[None, :] -> shape (n_non, n_neigh)
        if len(non_neighbors) > 0:
            strictly_closer = np.sum(d_non[:, None] < d_neigh[None, :], axis=0)
        else:
            strictly_closer = np.zeros(len(neighbors), dtype=np.int64)

        ranks = 1 + strictly_closer
        all_ranks.extend(ranks.tolist())

        # Average precision of u
        sorted_ranks = np.sort(ranks)
        k = np.arange(1, len(sorted_ranks) + 1, dtype=np.float64)
        precisions = k / (sorted_ranks + k - 1.0)
        node_aps.append(float(np.mean(precisions)))

    if len(all_ranks) == 0:
        return 0.0, 0.0

    mean_rank = float(np.mean(all_ranks))
    mean_ap = float(np.mean(node_aps))
    return mean_rank, mean_ap

# Step 29 - classical_mds
import numpy as np


def classical_mds(D, dim):
    D = np.asarray(D, dtype=np.float64)
    D2 = D ** 2

    # Double-centre D^2: B = -0.5 * J @ D2 @ J
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    # Eigendecomposition of symmetric matrix B
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort eigenvalues and eigenvectors in decreasing order
    idx = np.argsort(eigvals)[::-1][:dim]
    top_vals = np.maximum(eigvals[idx], 0.0)
    top_vecs = eigvecs[:, idx]

    return top_vecs * np.sqrt(top_vals)[None, :]


def average_distortion(D_emb, D_true):
    D_emb = np.asarray(D_emb, dtype=np.float64)
    D_true = np.asarray(D_true, dtype=np.float64)

    iu = np.triu_indices(D_true.shape[0], k=1)
    rho = D_emb[iu] / D_true[iu]
    c = np.mean(rho)
    return float(np.mean(np.abs(rho / c - 1.0)))

# Step 30 - graph_embeddings_report
import time
import numpy as np
from tinygrad import Tensor

GRAPH_EDM_TABLE = [
    {
        "method": "MDS",
        "encoder": "Lookup table",
        "target": "D",
        "decoder": "Euclidean distance",
        "loss": "Squared error",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "Laplacian eigenmaps",
        "encoder": "Lookup table",
        "target": "W",
        "decoder": "Euclidean distance",
        "loss": "Squared error",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "Poincare embedding",
        "encoder": "Lookup table",
        "target": "W",
        "decoder": "Poincare distance",
        "loss": "Tree softmax",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "DeepWalk / NetMF",
        "encoder": "Lookup table",
        "target": "s(W)",
        "decoder": "Inner product",
        "loss": "Cross-entropy / SE",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "Label propagation",
        "encoder": "Lookup table",
        "target": "W",
        "decoder": "identity",
        "loss": "Squared error",
        "alpha": 1,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "Label spreading",
        "encoder": "Lookup table",
        "target": "D^-1/2 W D^-1/2",
        "decoder": "identity",
        "loss": "Squared error",
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
    },
    {
        "method": "GCN classifier",
        "encoder": "GCN",
        "target": "None",
        "decoder": "None",
        "loss": "Cross-entropy",
        "alpha": 1,
        "beta": 0,
        "gamma": 1,
    },
    {
        "method": "GAE",
        "encoder": "GCN",
        "target": "W",
        "decoder": "Inner product",
        "loss": "Cross-entropy",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "VGAE",
        "encoder": "GCN",
        "target": "W",
        "decoder": "Inner product",
        "loss": "Cross-entropy",
        "alpha": 0,
        "beta": 1,
        "gamma": "KL",
    },
    {
        "method": "DGI",
        "encoder": "GCN",
        "target": "s(W, X)",
        "decoder": "Bilinear",
        "loss": "Contrastive",
        "alpha": 0,
        "beta": 1,
        "gamma": 0,
    },
    {
        "method": "Planetoid",
        "encoder": "Lookup + MLP",
        "target": "s(W)",
        "decoder": "Inner product",
        "loss": "Cross-entropy",
        "alpha": 1,
        "beta": 1,
        "gamma": 1,
    },
]


def graph_embeddings_report(seed=0, n=1000, scale=1.0, tree=(3, 5)):
    t0 = time.perf_counter()

    def S(steps):
        return max(2, int(steps * scale))

    # Main graph
    g = contextual_sbm(n=n, seed=seed)
    A = g["A"]
    X = g["X"]
    y = g["y"]
    P = g["P"]
    k = int(y.max()) + 1
    N = A.shape[0]

    # 1. graph stats
    degrees = A.sum(axis=1)
    graph_stats = {
        "edges": int(np.triu(A, 1).sum()),
        "homophily": float(edge_homophily(A, y)),
        "isolated": int(np.sum(degrees == 0)),
        "max_degree": int(degrees.max()),
    }

    # 2. flood (no degree correction, avg_degree=8.0, 2 labels per class)
    g_reg = contextual_sbm(n=n, avg_degree=8.0, degree_shape=0, seed=seed)
    A_reg, y_reg = g_reg["A"], g_reg["y"]
    k_reg = int(y_reg.max()) + 1
    labeled_reg, unlabeled_reg = few_labels(y_reg, 2, seed=seed)

    F_harm_reg = harmonic_labels(A_reg, labeled_reg, y_reg[labeled_reg], k_reg)
    F_norm_reg = class_mass_normalize(F_harm_reg, y_reg[labeled_reg], k_reg)
    F_spread_reg = label_spreading(A_reg, labeled_reg, y_reg[labeled_reg], k_reg)

    preds_harm_reg = F_harm_reg[unlabeled_reg].argmax(axis=1)
    counts_harm = np.bincount(preds_harm_reg, minlength=k_reg)
    largest_class_share = float(counts_harm.max() / len(unlabeled_reg))

    flood_stats = {
        "harmonic": float(node_accuracy(F_harm_reg, y_reg, unlabeled_reg)),
        "normalized": float(node_accuracy(F_norm_reg, y_reg, unlabeled_reg)),
        "spreading": float(node_accuracy(F_spread_reg, y_reg, unlabeled_reg)),
        "largest_class_share": largest_class_share,
    }

    # Precompute unsupervised / full-graph representations on main graph
    X_smooth = smoothed_features(A, X)
    S_dw = deepwalk_similarity(A, 5)

    ei, ej = np.nonzero(np.triu(A, 1))
    full_edges = np.column_stack((ei, ej)).astype(np.int64)
    Z_gae = train_gae({"A_train": A, "train": full_edges}, X, steps=S(200), seed=seed)["Z"]

    Z_dgi_untrained = train_dgi(A, X, steps=0, seed=seed)["Z"]
    Z_dgi = train_dgi(A, X, steps=S(200), seed=seed)["Z"]

    # 3. node_accuracy and 4. joint_beta
    node_acc = {}
    joint_beta = {}

    for c in (2, 10):
        labeled, unlabeled = few_labels(y, c, seed=seed)
        y_lab = y[labeled]

        F_harm = harmonic_labels(A, labeled, y_lab, k)
        F_norm = class_mass_normalize(F_harm, y_lab, k)
        F_spread = label_spreading(A, labeled, y_lab, k)

        pl_res = train_planetoid(
            S_dw, X, y, labeled, pre=S(500), joint=S(200), seed=seed
        )
        gcn_logits = train_gcn_classifier(A, X, y, labeled, beta=0.0, steps=S(200), seed=seed)

        node_acc[c] = {
            "features": float(linear_probe(X, y, labeled, unlabeled, seed=seed)),
            "harmonic": float(node_accuracy(F_harm, y, unlabeled)),
            "normalized": float(node_accuracy(F_norm, y, unlabeled)),
            "spreading": float(node_accuracy(F_spread, y, unlabeled)),
            "smoothed": float(linear_probe(X_smooth, y, labeled, unlabeled, seed=seed)),
            "dgi_untrained": float(linear_probe(Z_dgi_untrained, y, labeled, unlabeled, seed=seed)),
            "dgi": float(linear_probe(Z_dgi, y, labeled, unlabeled, seed=seed)),
            "gae": float(linear_probe(Z_gae, y, labeled, unlabeled, seed=seed)),
            "planetoid_two_stage": float(linear_probe(pl_res["Zc_pre"], y, labeled, unlabeled, seed=seed)),
            "planetoid": float(node_accuracy(pl_res["logits"], y, unlabeled)),
            "gcn": float(node_accuracy(gcn_logits, y, unlabeled)),
        }

        joint_beta[c] = {}
        for b_val in (0.0, 0.1, 1.0):
            if b_val == 0.0:
                logits_b = gcn_logits
            else:
                logits_b = train_gcn_classifier(A, X, y, labeled, beta=b_val, steps=S(200), seed=seed)
            joint_beta[c][b_val] = float(node_accuracy(logits_b, y, unlabeled))

    # 5. ari and zero_eigenvalues
    Z_em_0, evals_0 = laplacian_eigenmaps(A, 4, tau=0.0)
    zero_evals = int(np.sum(np.abs(evals_0) < 1e-9))

    Z_em_tau, _ = laplacian_eigenmaps(A, 4, tau=1.0)
    Zs_netmf, _ = netmf_embedding(A, T=5, dim=16)

    ari_stats = {
        "features": float(adjusted_rand_index(kmeans(X, 5, seed=seed), y)),
        "smoothed": float(adjusted_rand_index(kmeans(X_smooth, 5, seed=seed), y)),
        "eigenmaps": float(adjusted_rand_index(kmeans(Z_em_0, 5, seed=seed), y)),
        "eigenmaps_tau": float(adjusted_rand_index(kmeans(Z_em_tau, 5, seed=seed), y)),
        "netmf": float(adjusted_rand_index(kmeans(Zs_netmf, 5, seed=seed), y)),
        "dgi_untrained": float(adjusted_rand_index(kmeans(Z_dgi_untrained, 5, seed=seed), y)),
        "dgi": float(adjusted_rand_index(kmeans(Z_dgi, 5, seed=seed), y)),
        "gae": float(adjusted_rand_index(kmeans(Z_gae, 5, seed=seed), y)),
    }

    # 6. link_auc
    split = link_split(A, seed=seed)
    A_train = split["A_train"]
    test_pos = split["test_pos"]
    test_neg = split["test_neg"]

    Z_em_lp, _ = laplacian_eigenmaps(A_train, 16, tau=1.0)
    S_em_lp = -pairwise_distances(Z_em_lp, "euclidean") ** 2

    Zs_lp, Zt_lp = netmf_embedding(A_train, T=5, dim=16)

    Z_gae_lp = train_gae(split, X, loss="sampled", steps=S(200), seed=seed)["Z"]
    Z_gae_full_lp = train_gae(split, X, loss="full", steps=S(200), seed=seed)["Z"]
    Z_gae_full_uw_lp = train_gae(split, X, loss="full_unweighted", steps=S(200), seed=seed)["Z"]

    X_eye = np.eye(N, dtype=np.float32)
    Z_gae_nofeat = train_gae(split, X_eye, loss="sampled", steps=S(200), seed=seed)["Z"]

    split_nograph = dict(split)
    split_nograph["A_train"] = np.zeros_like(A_train)
    Z_gae_nograph = train_gae(split_nograph, X, loss="sampled", steps=S(200), seed=seed)["Z"]

    link_auc_stats = {
        "oracle": float(pair_auc(P, test_pos, test_neg)),
        "common_neighbors": float(pair_auc(A_train @ A_train, test_pos, test_neg)),
        "eigenmaps": float(pair_auc(S_em_lp, test_pos, test_neg)),
        "netmf": float(pair_auc(Zs_lp @ Zt_lp.T, test_pos, test_neg)),
        "gae": float(pair_auc(Z_gae_lp @ Z_gae_lp.T, test_pos, test_neg)),
        "gae_full": float(pair_auc(Z_gae_full_lp @ Z_gae_full_lp.T, test_pos, test_neg)),
        "gae_full_unweighted": float(pair_auc(Z_gae_full_uw_lp @ Z_gae_full_uw_lp.T, test_pos, test_neg)),
        "gae_no_features": float(pair_auc(Z_gae_nofeat @ Z_gae_nofeat.T, test_pos, test_neg)),
        "gae_no_graph": float(pair_auc(Z_gae_nograph @ Z_gae_nograph.T, test_pos, test_neg)),
    }

    # 7. vgae
    vgae_list = []
    for kl_w in (1.0, 100.0, 300.0, 1000.0):
        v_res = train_vgae(split, X, kl_weight=kl_w, steps=S(200), seed=seed)
        mu = v_res["mu"]
        kl_dim = v_res["kl_per_dim"]
        vgae_list.append({
            "kl_weight": float(kl_w),
            "auc": float(pair_auc(mu @ mu.T, test_pos, test_neg)),
            "kl_per_node": float(np.sum(kl_dim)),
            "active_dims": int(np.sum(kl_dim > 0.1)),
        })

    # 8. hetero
    g_het = contextual_sbm(n=n, avg_degree=8.0, homophily=0.05, degree_shape=0, seed=seed)
    A_het, X_het, y_het = g_het["A"], g_het["X"], g_het["y"]
    k_het = int(y_het.max()) + 1
    labeled_het, unlabeled_het = few_labels(y_het, 10, seed=seed)

    F_harm_het = harmonic_labels(A_het, labeled_het, y_het[labeled_het], k_het)
    F_spread_het = label_spreading(A_het, labeled_het, y_het[labeled_het], k_het)
    X_smooth_het = smoothed_features(A_het, X_het)
    gcn_het = train_gcn_classifier(A_het, X_het, y_het, labeled_het, steps=S(200), seed=seed)

    Z_em_het, _ = laplacian_eigenmaps(A_het, k_het - 1, tau=1.0)

    hetero_stats = {
        "homophily": float(edge_homophily(A_het, y_het)),
        "harmonic": float(node_accuracy(F_harm_het, y_het, unlabeled_het)),
        "spreading": float(node_accuracy(F_spread_het, y_het, unlabeled_het)),
        "features": float(linear_probe(X_het, y_het, labeled_het, unlabeled_het, seed=seed)),
        "smoothed": float(linear_probe(X_smooth_het, y_het, labeled_het, unlabeled_het, seed=seed)),
        "gcn": float(node_accuracy(gcn_het, y_het, unlabeled_het)),
        "ari_eigenmaps": float(adjusted_rand_index(kmeans(Z_em_het, k_het, seed=seed), y_het)),
        "ari_features": float(adjusted_rand_index(kmeans(X_het, k_het, seed=seed), y_het)),
    }

    # 9. tree
    A_tree, parent, level = balanced_tree(*tree)
    D_tree = tree_distances(parent)
    n_tree = A_tree.shape[0]

    poincare_res = {}
    for d in (2, 5):
        Z_p, _ = train_tree_embedding(
            A_tree, dim=d, geometry="poincare", steps=S(2000), burn=S(200), seed=seed
        )
        D_p = pairwise_distances(Z_p, "poincare")
        mr, mp = reconstruction_metrics(D_p, A_tree)
        dist_p = average_distortion(D_p, D_tree)
        root_radius = float(np.linalg.norm(Z_p[0]))
        root_dists = D_p[0]
        if np.std(root_dists) > 0 and np.std(level) > 0:
            corr = float(np.corrcoef(root_dists, level)[0, 1])
        else:
            corr = 0.0
        poincare_res[d] = {
            "mean_rank": float(mr),
            "map": float(mp),
            "distortion": float(dist_p),
            "root_radius": root_radius,
            "root_level_corr": corr,
        }

    euclidean_res = {}
    for d in (2, 5, 10):
        Z_e, _ = train_tree_embedding(
            A_tree, dim=d, geometry="euclidean", steps=S(2000), burn=S(200), seed=seed
        )
        D_e = pairwise_distances(Z_e, "euclidean")
        mr, mp = reconstruction_metrics(D_e, A_tree)
        dist_e = average_distortion(D_e, D_tree)
        euclidean_res[d] = {
            "mean_rank": float(mr),
            "map": float(mp),
            "distortion": float(dist_e),
        }

    mds_res = {}
    for d in (2, 5, 10):
        Z_m = classical_mds(D_tree, d)
        D_m = pairwise_distances(Z_m, "euclidean")
        mr, mp = reconstruction_metrics(D_m, A_tree)
        dist_m = average_distortion(D_m, D_tree)
        mds_res[d] = {
            "mean_rank": float(mr),
            "map": float(mp),
            "distortion": float(dist_m),
        }

    tree_stats = {
        "nodes": int(n_tree),
        "poincare": poincare_res,
        "euclidean": euclidean_res,
        "mds": mds_res,
    }

    elapsed = float(time.perf_counter() - t0)

    return {
        "graph": graph_stats,
        "flood": flood_stats,
        "node_accuracy": node_acc,
        "joint_beta": joint_beta,
        "ari": ari_stats,
        "zero_eigenvalues": zero_evals,
        "link_auc": link_auc_stats,
        "vgae": vgae_list,
        "hetero": hetero_stats,
        "tree": tree_stats,
        "table": GRAPH_EDM_TABLE,
        "seconds": elapsed,
    }


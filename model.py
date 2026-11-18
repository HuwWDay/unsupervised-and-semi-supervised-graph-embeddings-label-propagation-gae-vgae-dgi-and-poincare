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


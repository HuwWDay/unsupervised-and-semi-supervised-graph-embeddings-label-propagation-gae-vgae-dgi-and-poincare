"""
Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Unsupervised and Semi-Supervised Graph Embeddings: Label Propagation, GAE/VGAE, DGI and Poincare.

One seed is run end to end on a 1000-node degree-corrected contextual block model (five classes, 64 features,
homophily 0.8) and a 364-node balanced tree. The bracketed ranges are the same numbers over seeds 0-9,
measured on this image when the project was calibrated. Differences smaller than the spread are not findings.
"""
import time

RANGES = {
    "node_accuracy.2.features": "0.28-0.42, mean 0.35",
    "node_accuracy.2.harmonic": "0.21-0.40, mean 0.28",
    "node_accuracy.2.normalized": "0.68-0.82, mean 0.75",
    "node_accuracy.2.spreading": "0.51-0.73, mean 0.61",
    "node_accuracy.2.smoothed": "0.72-0.88, mean 0.82",
    "node_accuracy.2.dgi_untrained": "0.58-0.74, mean 0.66",
    "node_accuracy.2.dgi": "0.74-0.91, mean 0.85",
    "node_accuracy.2.gae": "0.54-0.78, mean 0.69",
    "node_accuracy.2.planetoid_two_stage": "0.43-0.71, mean 0.62",
    "node_accuracy.2.planetoid": "0.51-0.67, mean 0.59",
    "node_accuracy.2.gcn": "0.71-0.89, mean 0.83",
    "joint_beta.2.0.0": "0.71-0.89, mean 0.83",
    "joint_beta.2.0.1": "0.61-0.77, mean 0.72",
    "joint_beta.2.1.0": "0.53-0.73, mean 0.66",
    "node_accuracy.10.features": "0.44-0.55, mean 0.51",
    "node_accuracy.10.harmonic": "0.51-0.81, mean 0.69",
    "node_accuracy.10.normalized": "0.80-0.85, mean 0.83",
    "node_accuracy.10.spreading": "0.74-0.81, mean 0.79",
    "node_accuracy.10.smoothed": "0.90-0.93, mean 0.92",
    "node_accuracy.10.dgi_untrained": "0.82-0.90, mean 0.87",
    "node_accuracy.10.dgi": "0.91-0.94, mean 0.92",
    "node_accuracy.10.gae": "0.79-0.89, mean 0.84",
    "node_accuracy.10.planetoid_two_stage": "0.68-0.79, mean 0.75",
    "node_accuracy.10.planetoid": "0.58-0.73, mean 0.68",
    "node_accuracy.10.gcn": "0.91-0.94, mean 0.93",
    "joint_beta.10.0.0": "0.91-0.94, mean 0.93",
    "joint_beta.10.0.1": "0.87-0.90, mean 0.89",
    "joint_beta.10.1.0": "0.83-0.89, mean 0.86",
    "flood.harmonic": "0.32-0.59, mean 0.43",
    "flood.normalized": "0.83-0.94, mean 0.90",
    "flood.spreading": "0.74-0.90, mean 0.84",
    "flood.largest_class_share": "0.45-0.87, mean 0.69",
    "ari.features": "0.43-0.54, mean 0.46",
    "ari.smoothed": "0.66-0.89, mean 0.82",
    "ari.eigenmaps": "0.36-0.76, mean 0.60",
    "ari.eigenmaps_tau": "0.72-0.76, mean 0.75",
    "ari.netmf": "0.71-0.76, mean 0.74",
    "ari.dgi_untrained": "0.45-0.74, mean 0.55",
    "ari.dgi": "0.60-0.88, mean 0.82",
    "ari.gae": "0.78-0.83, mean 0.79",
    "link_auc.oracle": "0.85-0.88, mean 0.86",
    "link_auc.common_neighbors": "0.52-0.57, mean 0.55",
    "link_auc.eigenmaps": "0.57-0.62, mean 0.59",
    "link_auc.netmf": "0.67-0.70, mean 0.69",
    "link_auc.gae": "0.70-0.75, mean 0.72",
    "link_auc.gae_full": "0.69-0.75, mean 0.72",
    "link_auc.gae_full_unweighted": "0.48-0.57, mean 0.54",
    "link_auc.gae_no_features": "0.68-0.74, mean 0.71",
    "link_auc.gae_no_graph": "0.66-0.69, mean 0.67",
    "vgae.0.auc": "0.68-0.76, mean 0.71",
    "vgae.0.kl_per_node": "35.05-36.28, mean 35.68",
    "vgae.1.auc": "0.75-0.80, mean 0.77",
    "vgae.1.kl_per_node": "4.78-5.06, mean 4.93",
    "vgae.2.auc": "0.75-0.81, mean 0.77",
    "vgae.2.kl_per_node": "1.85-2.04, mean 1.98",
    "vgae.3.auc": "0.58-0.62, mean 0.60",
    "vgae.3.kl_per_node": "0.30-0.35, mean 0.33",
    "hetero.harmonic": "0.15-0.17, mean 0.16",
    "hetero.spreading": "0.15-0.18, mean 0.17",
    "hetero.features": "0.48-0.58, mean 0.54",
    "hetero.smoothed": "0.27-0.34, mean 0.31",
    "hetero.gcn": "0.27-0.33, mean 0.30",
    "hetero.ari_eigenmaps": "about 0 on all ten seeds",
    "hetero.ari_features": "0.27-0.56, mean 0.46",
    "tree.poincare.2.map": "0.93-0.95, mean 0.94",
    "tree.poincare.2.mean_rank": "2.38-4.18, mean 3.25",
    "tree.poincare.2.distortion": "0.17-0.19, mean 0.19",
    "tree.poincare.5.map": "0.97-0.99, mean 0.98",
    "tree.poincare.5.mean_rank": "1.38-2.15, mean 1.85",
    "tree.poincare.5.distortion": "0.19-0.20, mean 0.19",
    "tree.euclidean.2.map": "0.66-0.71, mean 0.68",
    "tree.euclidean.2.mean_rank": "5.09-7.03, mean 5.92",
    "tree.euclidean.2.distortion": "0.36-0.41, mean 0.39",
    "tree.euclidean.5.map": "0.75-0.77, mean 0.76",
    "tree.euclidean.5.mean_rank": "1.30-1.35, mean 1.33",
    "tree.euclidean.5.distortion": "0.20-0.22, mean 0.21",
    "tree.euclidean.10.map": "0.79-0.80, mean 0.79",
    "tree.euclidean.10.mean_rank": "1.21-1.22, mean 1.21",
    "tree.euclidean.10.distortion": "0.18-0.19, mean 0.18",
    "tree.mds.2.map": "0.04-0.04, mean 0.04",
    "tree.mds.2.mean_rank": "69.18-69.23, mean 69.19",
    "tree.mds.2.distortion": "0.55-0.55, mean 0.55",
    "tree.mds.5.map": "0.10-0.10, mean 0.10",
    "tree.mds.5.mean_rank": "22.11-22.17, mean 22.12",
    "tree.mds.5.distortion": "0.27-0.29, mean 0.28",
    "tree.mds.10.map": "0.16-0.16, mean 0.16",
    "tree.mds.10.mean_rank": "12.69-13.47, mean 13.31",
    "tree.mds.10.distortion": "0.15-0.15, mean 0.15",
    "tree.poincare.2.root_radius": "0.55-0.96, mean 0.82",
    "tree.poincare.2.root_level_corr": "0.75-0.85, mean 0.79"
}


def line(key, label, value, fmt=".3f"):
    print(f"   {label:50s} {value:{fmt}}   [{RANGES[key]}]")


def main() -> None:
    t0 = time.time()
    rep = graph_embeddings_report(seed=0)
    g = rep["graph"]
    print(f"Main graph: {g['edges']} edges, homophily {g['homophily']:.3f}, {g['isolated']} isolated nodes, largest degree {g['max_degree']}")

    print("\n1. Labels as embeddings: the flood at 1% labels (no degree correction, average degree 8, 2 labels per class)")
    f = rep["flood"]
    line("flood.harmonic", "harmonic solution (eq 23.19)", f["harmonic"])
    line("flood.largest_class_share", "share of unlabeled nodes in its largest class", f["largest_class_share"])
    line("flood.normalized", "same F after class mass normalisation", f["normalized"])
    line("flood.spreading", "label spreading (eq 23.20)", f["spreading"])

    print("\n2. Community recovery with no labels: adjusted Rand index of K-means")
    print(f"   zero eigenvalues of the Laplacian on this seed: {rep['zero_eigenvalues']} (over ten seeds: 1 to 2)")
    for key, label in (("features", "raw features"), ("eigenmaps", "Laplacian eigenmaps (eq 23.8)"), ("eigenmaps_tau", "eigenmaps, tau = 1"),
                       ("netmf", "NetMF: DeepWalk factorised, no walks (eq 23.17-23.18)"), ("smoothed", "two-hop smoothed features"),
                       ("gae", "GAE embedding"), ("dgi_untrained", "untrained DGI encoder"), ("dgi", "DGI embedding (eq 23.34)")):
        line(f"ari.{key}", label, rep["ari"][key])

    print("\n3. Link prediction on held-out edges: AUC")
    for key, label in (("oracle", "true edge probabilities (the ceiling)"), ("common_neighbors", "common neighbours"), ("eigenmaps", "eigenmaps"), ("netmf", "NetMF"),
                       ("gae", "GAE, negative sampling"), ("gae_full", "GAE, full loss with pos_weight (eq 23.31)"),
                       ("gae_full_unweighted", "GAE, full loss with no weighting: stalls at log 2"),
                       ("gae_no_features", "GAE, identity instead of features"), ("gae_no_graph", "GAE, no graph in the encoder")):
        line(f"link_auc.{key}", label, rep["link_auc"][key])
    print("   VGAE (eq 23.32-23.33). kl_weight = 1 is about the ELBO's own scale; 1000 = N forgets the 1/N.")
    for i, v in enumerate(rep["vgae"]):
        line(f"vgae.{i}.auc", f"kl_weight {v['kl_weight']:6.0f}: AUC", v["auc"])
        line(f"vgae.{i}.kl_per_node", f"kl_weight {v['kl_weight']:6.0f}: KL per node, nats ({v['active_dims']} of 16 dims above 0.1)", v["kl_per_node"])

    print("\n4. Node classification: accuracy on unlabeled nodes")
    for c in (2, 10):
        print(f"   {c} labels per class")
        a = rep["node_accuracy"][c]
        for key, label in (("features", "probe on raw features"), ("harmonic", "label propagation"), ("normalized", "label propagation + class mass normalisation"),
                           ("spreading", "label spreading"), ("planetoid_two_stage", "probe on context-only Zc (two-stage)"), ("planetoid", "Planetoid, joint (eq 23.35-23.36)"),
                           ("gae", "probe on GAE"), ("dgi_untrained", "probe on untrained DGI encoder"), ("dgi", "probe on DGI"),
                           ("smoothed", "probe on two-hop smoothing, nothing trained"), ("gcn", "GCN classifier")):
            line(f"node_accuracy.{c}.{key}", label, a[key])
        for b in (0.1, 1.0):
            line(f"joint_beta.{c}.{b}", f"GCN + reconstruction, beta = {b}", rep["joint_beta"][c][b])
    print("   Reconstruction lowered GCN accuracy on all ten seeds at both budgets.")
    print("   Joint Planetoid beat its own two-stage probe on 2 of 10 seeds at 2 labels and 0 of 10 at 10: the book's motivation did not reproduce here.")
    print("   DGI, GCN and plain smoothing are within the spread of each other at both budgets.")

    print("\n5. A graph that breaks the assumption: homophily " + f"{rep['hetero']['homophily']:.3f}, 10 labels per class, chance 0.2")
    for key, label in (("harmonic", "label propagation"), ("spreading", "label spreading"), ("smoothed", "probe on two-hop smoothing"), ("gcn", "GCN classifier"), ("features", "probe on raw features")):
        line(f"hetero.{key}", label, rep["hetero"][key])
    line("hetero.ari_eigenmaps", "ARI, eigenmaps", rep["hetero"]["ari_eigenmaps"])
    line("hetero.ari_features", "ARI, raw features", rep["hetero"]["ari_features"])

    print(f"\n6. Reconstructing a {rep['tree']['nodes']}-node tree (3-ary, depth 5)")
    print(f"   {'':24s} {'MAP':>6s} {'mean rank':>10s} {'distortion':>11s}")
    for geometry, dims in (("poincare", (2, 5)), ("euclidean", (2, 5, 10)), ("mds", (2, 5, 10))):
        for d in dims:
            e = rep["tree"][geometry][d]
            print(f"   {geometry + ' d=' + str(d):24s} {e['map']:6.3f} {e['mean_rank']:10.2f} {e['distortion']:11.3f}   [MAP {RANGES[f'tree.{geometry}.{d}.map']}]")
    line("tree.poincare.2.root_radius", "Poincare d=2: Euclidean norm of the root", rep["tree"]["poincare"][2]["root_radius"])
    line("tree.poincare.2.root_level_corr", "Poincare d=2: corr(distance to root, tree level)", rep["tree"]["poincare"][2]["root_level_corr"])
    print("   Poincare wins MAP at every dimension tried and mean rank at d=2. At d=5 Euclidean has the better mean rank.")
    print("   MDS at d=10 has the lowest distortion of all, yet poor local ranking. The loss does not put the root at the origin.")
    print("   MDS has no seed, but repeated eigenvalues can move its scores slightly from one machine to another.")

    print("\n7. The GraphEDM table (eq 23.4): L = alpha SUP + beta RECON + gamma REG")
    t = rep["tree"]; n = rep["node_accuracy"]; l = rep["link_auc"]
    measured = {"MDS": f"tree MAP {t['mds'][2]['map']:.2f} at d=2", "Laplacian eigenmaps": f"ARI {rep['ari']['eigenmaps_tau']:.2f} with tau",
                "Poincare embedding": f"tree MAP {t['poincare'][2]['map']:.2f} at d=2", "DeepWalk / NetMF": f"ARI {rep['ari']['netmf']:.2f}, AUC {l['netmf']:.2f}",
                "Label propagation": f"acc {n[10]['harmonic']:.2f} at 10/class", "Label spreading": f"acc {n[10]['spreading']:.2f} at 10/class",
                "GCN classifier": f"acc {n[10]['gcn']:.2f} at 10/class", "GAE": f"AUC {l['gae']:.2f}, ARI {rep['ari']['gae']:.2f}",
                "VGAE": f"AUC {rep['vgae'][1]['auc']:.2f} at kl_weight 100", "DGI": f"probe {n[10]['dgi']:.2f} at 10/class", "Planetoid": f"acc {n[10]['planetoid']:.2f} at 10/class"}
    print(f"   {'method':20s} {'encoder':16s} {'target s(W)':24s} {'decoder':21s} {'loss':24s} {'alpha':6s} {'beta':5s} {'gamma':13s} result")
    for row in rep["table"]:
        print(f"   {row['method']:20s} {row['encoder']:16s} {row['target']:24s} {row['decoder']:21s} {row['loss']:24s} {str(row['alpha']):6s} {str(row['beta']):5s} {str(row['gamma']):13s} {measured[row['method']]}")

    print(f"\nstudy finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()


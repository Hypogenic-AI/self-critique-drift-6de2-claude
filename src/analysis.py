"""Analysis of representation drift under self-critique.

Computes:
1. Cosine similarity and L2 distance between phase activations
2. PCA visualization of pre/post-critique representations
3. Linear probe accuracy for correctness prediction
4. Statistical tests comparing drift conditions
5. Behavioral outcome analysis
"""

import numpy as np
import json
import os
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_data():
    """Load behavioral results and activations."""
    # Try full results first, fall back to partial
    for suffix in ["", "_partial"]:
        results_path = os.path.join(RESULTS_DIR, f"behavioral_results{suffix}.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            break
    else:
        raise FileNotFoundError("No results found")

    activations = {}
    for phase in ["initial", "critique", "revised", "control"]:
        for suffix in ["", "_partial"]:
            path = os.path.join(RESULTS_DIR, f"activations_{phase}{suffix}.npy")
            if os.path.exists(path):
                activations[phase] = np.load(path)
                break

    print(f"Loaded {len(results)} results")
    for phase, arr in activations.items():
        print(f"  {phase}: {arr.shape}")
    return results, activations


def compute_drift_metrics(activations, results):
    """Compute cosine similarity and L2 distance between phases, per layer."""
    n_problems = activations["initial"].shape[0]
    n_layers = activations["initial"].shape[1]

    metrics = {
        "initial_to_revised": {"cosine": np.zeros((n_problems, n_layers)),
                                "l2": np.zeros((n_problems, n_layers))},
        "initial_to_control": {"cosine": np.zeros((n_problems, n_layers)),
                                "l2": np.zeros((n_problems, n_layers))},
        "initial_to_critique": {"cosine": np.zeros((n_problems, n_layers)),
                                 "l2": np.zeros((n_problems, n_layers))},
    }

    for i in range(n_problems):
        for l in range(n_layers):
            init = activations["initial"][i, l]
            rev = activations["revised"][i, l]
            ctrl = activations["control"][i, l]
            crit = activations["critique"][i, l]

            # Initial -> Revised (self-critique path)
            metrics["initial_to_revised"]["cosine"][i, l] = 1 - cosine(init, rev)
            metrics["initial_to_revised"]["l2"][i, l] = np.linalg.norm(init - rev)

            # Initial -> Control (re-prompt path)
            metrics["initial_to_control"]["cosine"][i, l] = 1 - cosine(init, ctrl)
            metrics["initial_to_control"]["l2"][i, l] = np.linalg.norm(init - ctrl)

            # Initial -> Critique
            metrics["initial_to_critique"]["cosine"][i, l] = 1 - cosine(init, crit)
            metrics["initial_to_critique"]["l2"][i, l] = np.linalg.norm(init - crit)

    return metrics


def analyze_drift_by_layer(metrics):
    """Summarize drift metrics per layer and compare critique vs control."""
    n_layers = metrics["initial_to_revised"]["cosine"].shape[1]

    layer_summary = []
    for l in range(n_layers):
        cos_critique = metrics["initial_to_revised"]["cosine"][:, l]
        cos_control = metrics["initial_to_control"]["cosine"][:, l]
        l2_critique = metrics["initial_to_revised"]["l2"][:, l]
        l2_control = metrics["initial_to_control"]["l2"][:, l]

        # Paired test: is critique drift different from control drift?
        t_cos, p_cos = stats.wilcoxon(cos_critique, cos_control, alternative="two-sided")
        t_l2, p_l2 = stats.wilcoxon(l2_critique, l2_control, alternative="two-sided")

        # Effect size (Cohen's d for paired)
        diff_cos = cos_critique - cos_control
        d_cos = np.mean(diff_cos) / (np.std(diff_cos) + 1e-10)
        diff_l2 = l2_critique - l2_control
        d_l2 = np.mean(diff_l2) / (np.std(diff_l2) + 1e-10)

        layer_summary.append({
            "layer": l,
            "cos_critique_mean": float(np.mean(cos_critique)),
            "cos_critique_std": float(np.std(cos_critique)),
            "cos_control_mean": float(np.mean(cos_control)),
            "cos_control_std": float(np.std(cos_control)),
            "cos_p_value": float(p_cos),
            "cos_cohens_d": float(d_cos),
            "l2_critique_mean": float(np.mean(l2_critique)),
            "l2_critique_std": float(np.std(l2_critique)),
            "l2_control_mean": float(np.mean(l2_control)),
            "l2_control_std": float(np.std(l2_control)),
            "l2_p_value": float(p_l2),
            "l2_cohens_d": float(d_l2),
        })

    return layer_summary


def train_correctness_probes(activations, results, layers_to_probe=None):
    """Train linear probes to predict correctness from representations.

    Compare probe accuracy on initial vs. revised vs. control activations.
    """
    n_problems = len(results)
    n_layers = activations["initial"].shape[1]

    if layers_to_probe is None:
        # Probe a subset of layers (early, middle, late)
        layers_to_probe = list(range(0, n_layers, max(1, n_layers // 8)))
        if (n_layers - 1) not in layers_to_probe:
            layers_to_probe.append(n_layers - 1)

    # Labels: ground truth correctness
    initial_labels = np.array([r["initial_correct"] for r in results], dtype=int)
    revised_labels = np.array([r["revised_correct"] for r in results], dtype=int)
    # For probe training, we use ground truth correctness as label
    # and test if representations are more linearly separable after critique
    gt_labels = initial_labels  # Use initial correctness as label for fair comparison

    probe_results = []
    for layer in layers_to_probe:
        for phase in ["initial", "revised", "control"]:
            X = activations[phase][:, layer, :]

            # Use correctness of this phase's answer as label
            if phase == "initial":
                y = initial_labels
            elif phase == "revised":
                y = revised_labels
            else:
                y = np.array([r["control_correct"] for r in results], dtype=int)

            # Skip if too few positive/negative examples
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validated logistic regression
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            try:
                scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc")
                probe_results.append({
                    "layer": layer,
                    "phase": phase,
                    "auc_mean": float(np.mean(scores)),
                    "auc_std": float(np.std(scores)),
                    "n_positive": int(y.sum()),
                    "n_negative": int(len(y) - y.sum()),
                })
            except Exception as e:
                print(f"  Probe failed for layer {layer}, phase {phase}: {e}")

    return probe_results


def pca_analysis(activations, results, layers_to_analyze=None):
    """PCA analysis of pre/post-critique representations.

    Visualize whether critique moves representations to a distinct subspace.
    """
    n_layers = activations["initial"].shape[1]
    if layers_to_analyze is None:
        # Analyze middle and late layers
        layers_to_analyze = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    pca_results = []
    for layer in layers_to_analyze:
        # Combine initial + revised for PCA
        X_init = activations["initial"][:, layer, :]
        X_rev = activations["revised"][:, layer, :]
        X_ctrl = activations["control"][:, layer, :]

        X_combined = np.vstack([X_init, X_rev, X_ctrl])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)

        pca = PCA(n_components=min(50, X_scaled.shape[0], X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        n = len(X_init)
        X_init_pca = X_pca[:n]
        X_rev_pca = X_pca[n:2*n]
        X_ctrl_pca = X_pca[2*n:]

        # Measure subspace overlap: how much variance is shared
        pca_init = PCA(n_components=10).fit(StandardScaler().fit_transform(X_init))
        pca_rev = PCA(n_components=10).fit(StandardScaler().fit_transform(X_rev))

        # Subspace angle between top principal components
        cos_angles = []
        for k in range(min(5, pca_init.components_.shape[0])):
            angle = np.abs(np.dot(pca_init.components_[k], pca_rev.components_[k]))
            cos_angles.append(float(angle))

        pca_results.append({
            "layer": layer,
            "variance_explained_3d": float(pca.explained_variance_ratio_[:3].sum()),
            "variance_explained_10d": float(pca.explained_variance_ratio_[:10].sum()),
            "subspace_cos_angles": cos_angles,
            "init_pca_2d": X_init_pca[:, :2].tolist(),
            "rev_pca_2d": X_rev_pca[:, :2].tolist(),
            "ctrl_pca_2d": X_ctrl_pca[:, :2].tolist(),
        })

    return pca_results


def behavioral_analysis(results):
    """Analyze behavioral outcomes: accuracy before/after critique."""
    n = len(results)
    initial_correct = [r["initial_correct"] for r in results]
    revised_correct = [r["revised_correct"] for r in results]
    control_correct = [r["control_correct"] for r in results]

    initial_acc = sum(initial_correct) / n
    revised_acc = sum(revised_correct) / n
    control_acc = sum(control_correct) / n

    # McNemar's test: initial vs revised
    # Count discordant pairs
    b = sum(1 for i in range(n) if initial_correct[i] and not revised_correct[i])  # correct -> wrong
    c = sum(1 for i in range(n) if not initial_correct[i] and revised_correct[i])  # wrong -> correct
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0
        mcnemar_p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    else:
        mcnemar_stat = 0
        mcnemar_p = 1.0

    # Answer change analysis
    changed = sum(1 for r in results if r["initial_answer"] != r["revised_answer"])
    improved = sum(1 for r in results if not r["initial_correct"] and r["revised_correct"])
    degraded = sum(1 for r in results if r["initial_correct"] and not r["revised_correct"])

    return {
        "n": n,
        "initial_accuracy": initial_acc,
        "revised_accuracy": revised_acc,
        "control_accuracy": control_acc,
        "answers_changed": changed,
        "improved": improved,
        "degraded": degraded,
        "mcnemar_p": float(mcnemar_p),
        "correct_to_wrong": b,
        "wrong_to_correct": c,
    }


def drift_outcome_correlation(metrics, results):
    """Correlate drift magnitude with behavioral outcome change."""
    n = len(results)
    n_layers = metrics["initial_to_revised"]["cosine"].shape[1]

    # Binary: did the answer change?
    answer_changed = np.array([
        1 if r["initial_answer"] != r["revised_answer"] else 0
        for r in results
    ])

    # Direction of change: improved (+1), degraded (-1), same (0)
    outcome_direction = np.array([
        (1 if r["revised_correct"] and not r["initial_correct"]
         else -1 if r["initial_correct"] and not r["revised_correct"]
         else 0)
        for r in results
    ])

    correlations = []
    # Analyze at selected layers
    for l in range(0, n_layers, max(1, n_layers // 8)):
        cos_drift = 1 - metrics["initial_to_revised"]["cosine"][:, l]  # Higher = more drift
        l2_drift = metrics["initial_to_revised"]["l2"][:, l]

        # Point-biserial correlation with answer change
        if answer_changed.sum() > 2 and answer_changed.sum() < len(answer_changed) - 2:
            r_cos, p_cos = stats.pointbiserialr(answer_changed, cos_drift)
            r_l2, p_l2 = stats.pointbiserialr(answer_changed, l2_drift)
        else:
            r_cos, p_cos, r_l2, p_l2 = 0, 1, 0, 1

        correlations.append({
            "layer": l,
            "cos_drift_vs_change_r": float(r_cos),
            "cos_drift_vs_change_p": float(p_cos),
            "l2_drift_vs_change_r": float(r_l2),
            "l2_drift_vs_change_p": float(p_l2),
        })

    return correlations


def plot_drift_by_layer(layer_summary, save_dir=PLOTS_DIR):
    """Plot cosine similarity and L2 distance across layers."""
    os.makedirs(save_dir, exist_ok=True)

    layers = [s["layer"] for s in layer_summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cosine similarity
    ax = axes[0]
    crit_means = [s["cos_critique_mean"] for s in layer_summary]
    ctrl_means = [s["cos_control_mean"] for s in layer_summary]
    crit_stds = [s["cos_critique_std"] for s in layer_summary]
    ctrl_stds = [s["cos_control_std"] for s in layer_summary]

    ax.plot(layers, crit_means, "b-o", label="Initial -> Revised (critique)", markersize=3)
    ax.fill_between(layers,
                    [m-s for m,s in zip(crit_means, crit_stds)],
                    [m+s for m,s in zip(crit_means, crit_stds)], alpha=0.2, color="blue")
    ax.plot(layers, ctrl_means, "r-s", label="Initial -> Control (re-prompt)", markersize=3)
    ax.fill_between(layers,
                    [m-s for m,s in zip(ctrl_means, ctrl_stds)],
                    [m+s for m,s in zip(ctrl_means, ctrl_stds)], alpha=0.2, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Representation Similarity Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L2 distance
    ax = axes[1]
    crit_l2 = [s["l2_critique_mean"] for s in layer_summary]
    ctrl_l2 = [s["l2_control_mean"] for s in layer_summary]
    crit_l2_std = [s["l2_critique_std"] for s in layer_summary]
    ctrl_l2_std = [s["l2_control_std"] for s in layer_summary]

    ax.plot(layers, crit_l2, "b-o", label="Initial -> Revised (critique)", markersize=3)
    ax.fill_between(layers,
                    [m-s for m,s in zip(crit_l2, crit_l2_std)],
                    [m+s for m,s in zip(crit_l2, crit_l2_std)], alpha=0.2, color="blue")
    ax.plot(layers, ctrl_l2, "r-s", label="Initial -> Control (re-prompt)", markersize=3)
    ax.fill_between(layers,
                    [m-s for m,s in zip(ctrl_l2, ctrl_l2_std)],
                    [m+s for m,s in zip(ctrl_l2, ctrl_l2_std)], alpha=0.2, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Distance")
    ax.set_title("Representation Distance Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "drift_by_layer.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved drift_by_layer.png")


def plot_pca(pca_results, results, save_dir=PLOTS_DIR):
    """Plot PCA projections of pre/post-critique representations."""
    os.makedirs(save_dir, exist_ok=True)

    initial_correct = np.array([r["initial_correct"] for r in results])
    revised_correct = np.array([r["revised_correct"] for r in results])

    for pca_res in pca_results:
        layer = pca_res["layer"]
        init_2d = np.array(pca_res["init_pca_2d"])
        rev_2d = np.array(pca_res["rev_pca_2d"])
        ctrl_2d = np.array(pca_res["ctrl_pca_2d"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Phases colored by condition
        ax = axes[0]
        ax.scatter(init_2d[:, 0], init_2d[:, 1], c="blue", alpha=0.4, s=20, label="Initial")
        ax.scatter(rev_2d[:, 0], rev_2d[:, 1], c="red", alpha=0.4, s=20, label="Revised")
        ax.scatter(ctrl_2d[:, 0], ctrl_2d[:, 1], c="green", alpha=0.4, s=20, label="Control")
        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title(f"Layer {layer}: Phase Comparison\n(Var explained: {pca_res['variance_explained_3d']:.1%} in 3 PCs)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Drift arrows from initial to revised, colored by outcome
        ax = axes[1]
        for i in range(len(init_2d)):
            color = "green" if revised_correct[i] and not initial_correct[i] else \
                    "red" if initial_correct[i] and not revised_correct[i] else \
                    "gray"
            ax.annotate("", xy=rev_2d[i], xytext=init_2d[i],
                       arrowprops=dict(arrowstyle="->", color=color, alpha=0.3, lw=0.5))
        ax.scatter(init_2d[:, 0], init_2d[:, 1], c="blue", alpha=0.5, s=15, label="Initial", zorder=5)
        ax.scatter(rev_2d[:, 0], rev_2d[:, 1], c="red", alpha=0.5, s=15, label="Revised", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Layer {layer}: Drift Arrows\n(Green=improved, Red=degraded, Gray=unchanged)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pca_layer_{layer}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved pca_layer_{layer}.png")


def plot_probe_results(probe_results, save_dir=PLOTS_DIR):
    """Plot probe AUC by layer and phase."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for phase, color in [("initial", "blue"), ("revised", "red"), ("control", "green")]:
        phase_data = [p for p in probe_results if p["phase"] == phase]
        if phase_data:
            layers = [p["layer"] for p in phase_data]
            aucs = [p["auc_mean"] for p in phase_data]
            stds = [p["auc_std"] for p in phase_data]
            ax.errorbar(layers, aucs, yerr=stds, fmt="-o", color=color,
                       label=f"{phase.capitalize()}", markersize=5, capsize=3)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUC (5-fold CV)")
    ax.set_title("Linear Probe Correctness Prediction by Phase")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "probe_auc.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved probe_auc.png")


def plot_behavioral(behavioral, save_dir=PLOTS_DIR):
    """Plot behavioral accuracy comparison."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    ax = axes[0]
    conditions = ["Initial", "Revised\n(after critique)", "Control\n(re-prompt)"]
    accs = [behavioral["initial_accuracy"], behavioral["revised_accuracy"],
            behavioral["control_accuracy"]]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(conditions, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("GSM8K Accuracy by Condition")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f"{acc:.1%}", ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Answer changes
    ax = axes[1]
    categories = ["Unchanged", "Improved\n(wrong->right)", "Degraded\n(right->wrong)"]
    counts = [behavioral["n"] - behavioral["improved"] - behavioral["degraded"],
              behavioral["improved"], behavioral["degraded"]]
    colors2 = ["#CCCCCC", "#55A868", "#C44E52"]
    ax.bar(categories, counts, color=colors2, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Answer Change Analysis After Self-Critique")
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(c), ha="center", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "behavioral.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved behavioral.png")


def plot_drift_distribution(metrics, results, save_dir=PLOTS_DIR):
    """Plot distribution of drift for changed vs unchanged answers."""
    os.makedirs(save_dir, exist_ok=True)

    n_layers = metrics["initial_to_revised"]["cosine"].shape[1]
    mid_layer = n_layers // 2
    late_layer = n_layers - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, layer, title in [(axes[0], mid_layer, f"Middle Layer ({mid_layer})"),
                              (axes[1], late_layer, f"Last Layer ({late_layer})")]:
        cos_drift = 1 - metrics["initial_to_revised"]["cosine"][:, layer]
        changed = [1 if r["initial_answer"] != r["revised_answer"] else 0 for r in results]

        drift_changed = cos_drift[np.array(changed) == 1]
        drift_unchanged = cos_drift[np.array(changed) == 0]

        if len(drift_changed) > 0:
            ax.hist(drift_changed, bins=20, alpha=0.6, color="red", label=f"Changed (n={len(drift_changed)})", density=True)
        if len(drift_unchanged) > 0:
            ax.hist(drift_unchanged, bins=20, alpha=0.6, color="blue", label=f"Unchanged (n={len(drift_unchanged)})", density=True)
        ax.set_xlabel("Cosine Drift (1 - similarity)")
        ax.set_ylabel("Density")
        ax.set_title(f"{title}: Drift by Answer Change")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "drift_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved drift_distribution.png")


def run_full_analysis():
    """Run all analyses and save results."""
    results, activations = load_data()

    # 1. Behavioral analysis
    print("\n=== Behavioral Analysis ===")
    behavioral = behavioral_analysis(results)
    print(f"Initial accuracy: {behavioral['initial_accuracy']:.3f}")
    print(f"Revised accuracy: {behavioral['revised_accuracy']:.3f}")
    print(f"Control accuracy: {behavioral['control_accuracy']:.3f}")
    print(f"Answers changed: {behavioral['answers_changed']}/{behavioral['n']}")
    print(f"Improved: {behavioral['improved']}, Degraded: {behavioral['degraded']}")
    print(f"McNemar's p-value: {behavioral['mcnemar_p']:.4f}")

    # 2. Drift metrics
    print("\n=== Drift Metrics ===")
    metrics = compute_drift_metrics(activations, results)
    layer_summary = analyze_drift_by_layer(metrics)

    # Print a few key layers
    n_layers = len(layer_summary)
    for idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        s = layer_summary[idx]
        print(f"Layer {s['layer']:2d}: cos(init,rev)={s['cos_critique_mean']:.4f} "
              f"cos(init,ctrl)={s['cos_control_mean']:.4f} "
              f"p={s['cos_p_value']:.4f} d={s['cos_cohens_d']:.3f}")

    # Count significant layers (before FDR correction)
    sig_layers = sum(1 for s in layer_summary if s["cos_p_value"] < 0.05)
    print(f"\nLayers with significant critique vs control difference: {sig_layers}/{n_layers}")

    # FDR correction
    p_values = [s["cos_p_value"] for s in layer_summary]
    from statsmodels.stats.multitest import multipletests
    try:
        reject, corrected_p, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)
        sig_fdr = sum(reject)
        print(f"After FDR correction: {sig_fdr}/{n_layers} significant")
        for i, s in enumerate(layer_summary):
            s["cos_p_fdr"] = float(corrected_p[i])
            s["cos_sig_fdr"] = bool(reject[i])
    except ImportError:
        print("statsmodels not available, skipping FDR correction")

    # 3. Probing analysis
    print("\n=== Probing Analysis ===")
    probe_results = train_correctness_probes(activations, results)
    for p in probe_results:
        print(f"Layer {p['layer']:2d} [{p['phase']:8s}]: AUC = {p['auc_mean']:.3f} +/- {p['auc_std']:.3f}")

    # 4. PCA analysis
    print("\n=== PCA Analysis ===")
    pca_results = pca_analysis(activations, results)
    for p in pca_results:
        print(f"Layer {p['layer']:2d}: var_3d={p['variance_explained_3d']:.3f}, "
              f"subspace_angles={[f'{a:.3f}' for a in p['subspace_cos_angles'][:3]]}")

    # 5. Drift-outcome correlation
    print("\n=== Drift-Outcome Correlation ===")
    correlations = drift_outcome_correlation(metrics, results)
    for c in correlations:
        print(f"Layer {c['layer']:2d}: r_cos={c['cos_drift_vs_change_r']:.3f} "
              f"(p={c['cos_drift_vs_change_p']:.4f})")

    # 6. Generate plots
    print("\n=== Generating Plots ===")
    plot_drift_by_layer(layer_summary)
    plot_pca(pca_results, results)
    plot_probe_results(probe_results)
    plot_behavioral(behavioral)
    plot_drift_distribution(metrics, results)

    # 7. Save all analysis results
    all_results = {
        "behavioral": behavioral,
        "layer_summary": layer_summary,
        "probe_results": probe_results,
        "pca_results": [{k: v for k, v in p.items() if k not in ["init_pca_2d", "rev_pca_2d", "ctrl_pca_2d"]}
                        for p in pca_results],
        "correlations": correlations,
    }
    with open(os.path.join(RESULTS_DIR, "analysis_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULTS_DIR}/analysis_results.json")

    return all_results


if __name__ == "__main__":
    run_full_analysis()

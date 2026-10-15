import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Social Cluster Recommendation Engine for Co-Mutating Genes — MSK Breast 2018

*(October 2026)*

The earlier notebooks in this repo found gene co-occurrence pairs with Random Forest feature stability and Fisher's exact tests. That approach scores each pair on its own. This notebook treats the cohort as a **social network**: patients are users and mutated genes are the items they "interact with". It then uses the algorithms that social platforms and recommender systems use on that kind of data:

| Social / recommender concept | Genomic analogue in this notebook |
|---|---|
| User × item interaction matrix | Patient × gene binary mutation matrix |
| "Friends" / co-engagement graph | Gene–gene co-mutation network |
| Degree-preserving random graph null (configuration model) | **Curveball** null: shuffles mutations while preserving every patient's mutation burden *and* every gene's frequency |
| Community detection (Louvain, label propagation) | **Gene clusters that mutate together** |
| Link prediction scores (Jaccard, Adamic–Adar) | Pairwise co-mutation similarity |
| "People who bought X also bought Y" (item-kNN) | Genes recommended from a patient's existing mutations |
| "People like you" (user-kNN) | Mutation profiles of similar patients |
| Who-To-Follow random walk (personalized PageRank) | Random walk with restart over the patient–gene bipartite graph |
| Matrix factorization | NMF latent "mutational programs" |

**Why the null model matters.** Raw co-occurrence counts mostly track gene frequency (PIK3CA and TP53 co-occur with everything) and tumor mutation burden (hypermutated tumors co-mutate everything). Social-network analysis has the same problem: popular users get linked to everyone. The standard fix in both fields is to compare the observed graph against random graphs that have the same degree sequence. That is what the curveball algorithm does (Strona et al. 2014; also the null behind DISCOVER, Canisius et al. 2016).

**Questions answered:**
1. Which genes mutate together (or avoid each other) more than burden and frequency alone would predict?
2. What clusters (communities) do these genes form, and how stable are they?
3. Do the clusters line up with clinical subtypes?
4. Given a patient's known mutations, can a recommender rank their *other* mutated genes better than "just guess the most common genes"?
""")

md("## 1. Setup and Configuration")
code(r"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 120)
sns.set_theme(style="whitegrid", context="notebook")

SEED = 677                 # reproducible: every random step below draws from this
MIN_FREQ_NETWORK = 0.01    # genes mutated in >=1% of patients enter the co-mutation network
MIN_PATIENTS_RECSYS = 5    # genes mutated in >=5 patients are candidate recommendations
N_NULL = 2000              # curveball null samples for the main network
N_BOOT = 30                # patient bootstraps for community stability
N_NULL_BOOT = 200          # curveball null samples inside each bootstrap / CV fold
FDR = 0.10                 # Benjamini-Hochberg threshold for calling an edge
EDGE_P = 0.01              # nominal co-occurrence p for an edge in the (exploratory) clustering network
HYPERMUT_MAX_GENES = 20    # patients with more mutated core genes are treated as hypermutated and excluded
INCLUDE_CNA = False        # True adds ERBB2_AMP-style amplification / deep-deletion "genes"

rng = np.random.default_rng(SEED)
""")

md(r"""
## 2. Build the Patient × Gene Interaction Matrix

Choices made here (each one avoids a known artifact):

- **One sample per patient.** 144 patients have 2–4 samples. Counting all of them would let one patient's tumor "vote" several times. The primary sample is kept when one exists.
- **Non-silent coding mutations only.** Silent, intronic, UTR, flank and splice-region calls are dropped.
- **Hypermutated tumors excluded.** About 2% of tumors carry more than 20 mutated core genes (roughly ≥20 mutations/Mb on this panel, a common hypermutation cutoff). Their mutations land on large genes because of mutational processes (POLE, MMR deficiency, APOBEC), not selection. Even a burden-preserving null cannot correct for that gene-length bias.
- **Common panel genes only.** The cohort was sequenced on three MSK-IMPACT panels (341, 410, 468 genes). A gene missing from the smaller panel looks "not mutated" in those patients, which creates false mutual exclusivity. The CNA matrix is zero-filled, so it can't be used to infer coverage. As a conservative proxy, this keeps genes that are on IMPACT410 *and* observed mutated in at least one IMPACT341 sample.
""")
code(r"""
NONSILENT = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
             "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Translation_Start_Site", "Nonstop_Mutation"}

df_mut = pd.read_csv("breast_msk_2018/data_mutations.txt", sep="\t", low_memory=False)
df_sample = pd.read_csv("breast_msk_2018/data_clinical_sample.txt", sep="\t", skiprows=4)
df_patient = pd.read_csv("breast_msk_2018/data_clinical_patient.txt", sep="\t", skiprows=4)
df_panel = pd.read_csv("breast_msk_2018/data_gene_panel_matrix.txt", sep="\t")

df_mut = df_mut[df_mut["Variant_Classification"].isin(NONSILENT)]
df_sample = df_sample.merge(df_panel[["SAMPLE_ID", "mutations"]].rename(columns={"mutations": "PANEL"}), on="SAMPLE_ID")

# One sample per patient, primary preferred
df_sample["_not_primary"] = (df_sample["SAMPLE_TYPE"] != "Primary").astype(int)
samples = (df_sample.sort_values(["PATIENT_ID", "_not_primary", "SAMPLE_ID"])
                    .drop_duplicates("PATIENT_ID")
                    .drop(columns="_not_primary")
                    .set_index("SAMPLE_ID"))

# Genes testable on every panel (conservative proxy, see markdown above)
with open("misc_data/data_gene_panel_impact410.txt") as fh:
    genes_410 = set(fh.read().split("gene_list:")[1].split())
im341_samples = set(df_sample.loc[df_sample["PANEL"] == "IMPACT341", "SAMPLE_ID"])
genes_seen_341 = set(df_mut.loc[df_mut["Tumor_Sample_Barcode"].isin(im341_samples), "Hugo_Symbol"])
core_genes = sorted(genes_410 & genes_seen_341)

mut = df_mut[df_mut["Tumor_Sample_Barcode"].isin(samples.index) & df_mut["Hugo_Symbol"].isin(core_genes)]
X_full = (pd.crosstab(mut["Tumor_Sample_Barcode"], mut["Hugo_Symbol"]).clip(upper=1)
            .reindex(index=samples.index, fill_value=0))

if INCLUDE_CNA:
    cna = pd.read_csv("breast_msk_2018/data_cna.txt", sep="\t").set_index("Hugo_Symbol")
    cna = cna.loc[cna.index.intersection(core_genes), cna.columns.intersection(samples.index)].T
    amp = (cna == 2).add_suffix("_AMP").astype(int)
    dele = (cna == -2).add_suffix("_DEL").astype(int)
    X_full = X_full.join(amp, how="left").join(dele, how="left").fillna(0).astype(int)
    X_full = X_full.loc[:, X_full.sum() > 0]

hypermutated = X_full.sum(axis=1) > HYPERMUT_MAX_GENES
X_full = X_full[~hypermutated]
X_full = X_full.loc[:, X_full.sum() > 0]
samples = samples.loc[X_full.index]

print(f"Hypermutated patients excluded (> {HYPERMUT_MAX_GENES} mutated core genes): {hypermutated.sum()}")
print(f"Patients (one sample each): {X_full.shape[0]}")
print(f"Core genes/features:        {X_full.shape[1]}")
print(f"Patients with 0 core mutations: {(X_full.sum(axis=1) == 0).sum()}")
print(f"Median mutated genes / patient: {X_full.sum(axis=1).median():.0f}  (max {X_full.sum(axis=1).max()})")
""")

code(r"""
freq = X_full.sum().sort_values(ascending=False)
burden = X_full.sum(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
(freq.head(30) / len(X_full) * 100).plot.bar(ax=axes[0], color="#4C72B0")
axes[0].set_ylabel("% of patients mutated")
axes[0].set_title("Most frequently mutated core genes")
axes[1].hist(burden, bins=range(0, burden.max() + 2), color="#55A868")
axes[1].set_xlabel("Mutated core genes per patient")
axes[1].set_ylabel("Patients")
axes[1].set_title("Mutation burden (the 'degree' of each patient)")
axes[1].set_yscale("log")
plt.tight_layout()
plt.show()
""")

md(r"""
The burden histogram is the reason a naïve co-occurrence count is misleading. The distribution is heavy-tailed, so a few hypermutated tumors are "connected" to dozens of genes at once. In social-network terms they behave like spam accounts that follow everyone.

## 3. Degree-Preserving Null Model (Curveball Algorithm)

The curveball algorithm (Strona et al., *Nat. Commun.* 2014) produces random binary matrices that keep **exactly** the same row sums (each patient's burden) and column sums (each gene's frequency). Each "trade" picks two patients and randomly redistributes the genes that only one of them carries. Repeating this many times samples uniformly from all matrices with those margins.

For every gene pair we compare the observed number of co-mutated patients with its null distribution:
- **z-score** = (observed − null mean) / null SD
- **empirical one-sided p** for co-occurrence (observed ≥ null) and for mutual exclusivity (observed ≤ null)
- **Benjamini–Hochberg q-values** across all pairs

A purely empirical p-value cannot go below 1/(N_NULL+1). With thousands of pairs tested, that floor makes FDR control impossible. Tail p-values therefore come from a binomial moment-matched to each pair's null mean and variance. The cell after the statistics checks that these agree with the raw empirical p-values wherever the empirical p is resolvable.
""")
code(r"""
def curveball_sampler(A, n_samples, burn_in_sweeps=50, thin_sweeps=1, rng=rng):
    # Yield co-occurrence matrices (genes x genes) from margin-preserving random matrices.
    rows = [set(np.flatnonzero(r)) for r in A]
    n_rows, n_cols = A.shape
    active = [i for i, r in enumerate(rows) if 0 < len(r) < n_cols]

    def sweep():
        for _ in range(len(active) // 2):
            i, j = rng.choice(active, 2, replace=False)
            a, b = rows[i], rows[j]
            a_only, b_only = a - b, b - a
            if not a_only or not b_only:
                continue
            pool = np.array(list(a_only | b_only))
            rng.shuffle(pool)
            k = len(a_only)
            shared = a & b
            rows[i] = shared | set(pool[:k].tolist())
            rows[j] = shared | set(pool[k:].tolist())

    for _ in range(burn_in_sweeps):
        sweep()
    for _ in range(n_samples):
        for _ in range(thin_sweeps):
            sweep()
        r_idx = np.repeat(np.arange(n_rows), [len(r) for r in rows])
        c_idx = np.fromiter((c for r in rows for c in r), dtype=int)
        M = csr_matrix((np.ones(len(c_idx)), (r_idx, c_idx)), shape=A.shape)
        yield (M.T @ M).toarray()


def comutation_stats(X, n_null, rng=rng):
    # Observed vs curveball-null co-occurrence for every gene pair in X (patients x genes DataFrame).
    A = X.values.astype(np.int8)
    obs = (A.T.astype(np.int32) @ A).astype(float)
    g = A.shape[1]
    s1 = np.zeros((g, g)); s2 = np.zeros((g, g))
    n_ge = np.zeros((g, g)); n_le = np.zeros((g, g))
    for C in curveball_sampler(A, n_null, rng=rng):
        s1 += C; s2 += C ** 2
        n_ge += C >= obs; n_le += C <= obs
    mean = s1 / n_null
    var = np.maximum(s2 / n_null - mean ** 2, 1e-12)
    z = (obs - mean) / np.sqrt(var)
    # Tail p-values from a binomial moment-matched to each pair's null (mean, variance).
    # The null count is a sum of near-independent patient-level Bernoullis (variance < mean),
    # for which a matched binomial is the standard approximation; it resolves tails far below 1/N_NULL.
    p_bin = np.clip(1 - var / np.maximum(mean, 1e-12), 1e-6, 1)
    n_bin = np.maximum(np.round(mean / p_bin), np.ceil(obs)).astype(int)
    p_bin = np.clip(mean / np.maximum(n_bin, 1), 1e-12, 1)
    p_co = stats.binom.sf(obs - 1, n_bin, p_bin)
    p_ex = stats.binom.cdf(obs, n_bin, p_bin)
    emp_co = (n_ge + 1) / (n_null + 1)
    emp_ex = (n_le + 1) / (n_null + 1)
    return obs, mean, z, p_co, p_ex, emp_co, emp_ex
""")

code(r"""
net_genes = freq[freq >= MIN_FREQ_NETWORK * len(X_full)].index.tolist()
X_net = X_full[net_genes]
print(f"Genes in co-mutation network (>= {MIN_FREQ_NETWORK:.0%} of patients): {len(net_genes)}")

obs, exp_null, Z, P_co, P_ex, E_co, E_ex = comutation_stats(X_net, N_NULL)

iu = np.triu_indices(len(net_genes), k=1)
n = len(X_net)
nA = X_net.sum().values
pairs = pd.DataFrame({
    "Gene_A": np.array(net_genes)[iu[0]],
    "Gene_B": np.array(net_genes)[iu[1]],
    "n_A": nA[iu[0]], "n_B": nA[iu[1]],
    "Observed": obs[iu].astype(int),
    "Expected_null": exp_null[iu].round(2),
    "z": Z[iu].round(2),
    "p_cooccur": P_co[iu],
    "p_exclusive": P_ex[iu],
    "p_emp_cooccur": E_co[iu],
    "p_emp_exclusive": E_ex[iu],
})
pairs["q_cooccur"] = multipletests(pairs["p_cooccur"], method="fdr_bh")[1]
pairs["q_exclusive"] = multipletests(pairs["p_exclusive"], method="fdr_bh")[1]

# Classic social-network link-prediction scores, for comparison with the null-adjusted z
union = pairs["n_A"] + pairs["n_B"] - pairs["Observed"]
pairs["Jaccard"] = (pairs["Observed"] / union).round(3)
pairs["Lift"] = (pairs["Observed"] * n / (pairs["n_A"] * pairs["n_B"])).round(2)
inv_log_burden = 1 / np.log(np.maximum(X_net.sum(axis=1).values, 2))
AA = (X_net.values.T * inv_log_burden) @ X_net.values   # Adamic-Adar: shared patients weighted by 1/log(burden)
pairs["Adamic_Adar"] = AA[iu].round(2)

print(f"Gene pairs tested: {len(pairs)}")
print(f"Co-occurring  pairs at q < {FDR}: {(pairs.q_cooccur < FDR).sum()}")
print(f"Mutually exclusive pairs at q < {FDR}: {(pairs.q_exclusive < FDR).sum()}")
print(f"Smallest empirical p with N_NULL={N_NULL}: {1/(N_NULL+1):.1e}  (moment-matched p is used for FDR)")
""")

code(r"""
# Calibration: moment-matched vs empirical curveball p-values (only where empirical p is resolvable)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
floor = 1 / (N_NULL + 1)
for ax, a, b, t in [(axes[0], "p_cooccur", "p_emp_cooccur", "Co-occurrence"), (axes[1], "p_exclusive", "p_emp_exclusive", "Exclusivity")]:
    ok = pairs[b] > 5 * floor
    ax.scatter(pairs.loc[ok, b], pairs.loc[ok, a], s=6, alpha=0.4)
    ax.plot([floor, 1], [floor, 1], "k--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("empirical curveball p"); ax.set_ylabel("moment-matched binomial p")
    rho = stats.spearmanr(pairs.loc[ok, a], pairs.loc[ok, b])[0]
    ax.set_title(f"{t}: Spearman rho = {rho:.3f}")
plt.tight_layout()
plt.show()
""")

md("### Strongest co-occurring pairs (mutate together more than burden and frequency predict)")
code(r"""
cols = ["Gene_A", "Gene_B", "n_A", "n_B", "Observed", "Expected_null", "z", "q_cooccur", "Lift", "Jaccard", "Adamic_Adar"]
top_co = pairs[pairs.q_cooccur < FDR].sort_values("z", ascending=False)
top_co[cols].head(25)
""")

md("### Strongest mutually exclusive pairs (mutate together *less* than expected)")
code(r"""
cols_ex = ["Gene_A", "Gene_B", "n_A", "n_B", "Observed", "Expected_null", "z", "q_exclusive"]
top_ex = pairs[pairs.q_exclusive < FDR].sort_values("z")
top_ex[cols_ex].head(25)
""")

md(r"""
### Raw popularity versus null-adjusted scores

This shows how much the ranking changes once the null model is applied. Raw counts and Jaccard reward common genes. The z-score asks only whether the overlap is *surprising*.
""")
code(r"""
rank_corr = pairs[["Observed", "Jaccard", "Lift", "Adamic_Adar", "z"]].corr(method="spearman")["z"].drop("z")
print("Spearman correlation of each heuristic score with the null-adjusted z:")
print(rank_corr.round(2).to_string())

top_raw = set(pairs.nlargest(20, "Observed")[["Gene_A", "Gene_B"]].apply(tuple, axis=1))
top_z = set(pairs.nlargest(20, "z")[["Gene_A", "Gene_B"]].apply(tuple, axis=1))
print(f"\nOverlap between top-20 pairs by raw count and top-20 by z: {len(top_raw & top_z)} / 20")
""")

md(r"""
## 4. Community Detection — Gene Clusters That Mutate Together

Only 11 pairs pass FDR, which is too few edges to show cluster structure. The clustering network is therefore **exploratory**: it keeps an edge where co-occurrence beats the null at nominal p < `EDGE_P` (0.01), weighted by z. Noise is controlled in the next step by checking whether each cluster survives patient bootstraps. In the network plot, FDR-significant edges are drawn in bold. Three standard social community-detection algorithms are run on it and compared:

- **Louvain** modularity maximization (Blondel et al. 2008), the default on most social-graph platforms
- **Greedy modularity** (Clauset–Newman–Moore)
- **Label propagation** (Raghavan et al. 2007), which works like rumor spreading: each node adopts its neighbors' majority label

Agreement between methods is measured with adjusted Rand index (ARI) and normalized mutual information (NMI). Genes with no significant edge are left as singletons and excluded from the comparison.
""")
code(r"""
def build_network(pairs_df, genes, edge_p=EDGE_P):
    G = nx.Graph()
    G.add_nodes_from(genes)
    sig = pairs_df[(pairs_df["p_cooccur"] < edge_p) & (pairs_df["z"] > 0)]
    G.add_weighted_edges_from(sig[["Gene_A", "Gene_B", "z"]].itertuples(index=False, name=None))
    return G


def partition_labels(communities, nodes):
    lab = {g: i for i, c in enumerate(communities) for g in c}
    return np.array([lab[g] for g in nodes])


G = build_network(pairs, net_genes)
connected = [g for g in G.nodes if G.degree(g) > 0]
H = G.subgraph(connected)
print(f"Network: {H.number_of_nodes()} connected genes, {H.number_of_edges()} co-mutation edges at p < {EDGE_P} "
      f"({len(net_genes) - len(connected)} isolated genes)")

methods = {
    "Louvain": nx.community.louvain_communities(H, weight="weight", seed=SEED),
    "Greedy modularity": nx.community.greedy_modularity_communities(H, weight="weight"),
    "Label propagation": list(nx.community.asyn_lpa_communities(H, weight="weight", seed=SEED)),
}
labels = {k: partition_labels(v, connected) for k, v in methods.items()}

summary = pd.DataFrame({
    k: {"n_communities": len(v),
        "modularity": nx.community.modularity(H, v, weight="weight"),
        "largest": max(len(c) for c in v)}
    for k, v in methods.items()
}).T
names = list(methods)
agree = pd.DataFrame([(a, b, adjusted_rand_score(labels[a], labels[b]), normalized_mutual_info_score(labels[a], labels[b]))
                      for i, a in enumerate(names) for b in names[i + 1:]],
                     columns=["Method_A", "Method_B", "ARI", "NMI"]).round(3)
display(summary.round(3))
display(agree)
""")

md(r"""
### Stability: do the clusters survive resampling patients?

A cluster that disappears when you resample the cohort is noise. The step below repeats the whole pipeline (curveball null → edges → Louvain) on **N_BOOT** bootstrap resamples of patients and records how often each gene pair lands in the same community. The result is a *consensus matrix*.
""")
code(r"""
consensus = pd.DataFrame(0.0, index=connected, columns=connected)
both_present = pd.DataFrame(0.0, index=connected, columns=connected)

for b in range(N_BOOT):
    idx = rng.choice(len(X_net), len(X_net), replace=True)
    Xb = X_net.iloc[idx].reset_index(drop=True)
    Xb = Xb.loc[:, Xb.sum() > 0]
    gb = Xb.columns.tolist()
    ob, _, zb, pcb, *_ = comutation_stats(Xb, N_NULL_BOOT)
    iub = np.triu_indices(len(gb), k=1)
    pb = pd.DataFrame({"Gene_A": np.array(gb)[iub[0]], "Gene_B": np.array(gb)[iub[1]],
                       "z": zb[iub], "p_cooccur": pcb[iub]})
    Gb = build_network(pb, gb)
    Gb = Gb.subgraph([g for g in Gb if Gb.degree(g) > 0])
    lab = {g: i for i, c in enumerate(nx.community.louvain_communities(Gb, weight="weight", seed=SEED + b)) for g in c}
    present = [g for g in connected if g in lab]
    v = pd.Series({g: lab[g] for g in present})
    same = (v.values[:, None] == v.values[None, :]).astype(float)
    consensus.loc[present, present] += same
    both_present.loc[present, present] += 1

consensus = (consensus / both_present.replace(0, np.nan)).fillna(0)

louvain = sorted(methods["Louvain"], key=len, reverse=True)
community_of = {g: i + 1 for i, c in enumerate(louvain) for g in c}

rows = []
for i, c in enumerate(louvain, start=1):
    c = sorted(c, key=lambda g: -freq[g])
    sub = consensus.loc[c, c].values
    within = sub[np.triu_indices(len(c), k=1)].mean() if len(c) > 1 else np.nan
    rows.append({"Community": i, "Size": len(c), "Bootstrap co-assignment": round(within, 2),
                 "Genes (most frequent first)": ", ".join(c)})
communities_df = pd.DataFrame(rows)
communities_df
""")

md(r"""
*Bootstrap co-assignment* is the average fraction of resamples in which two genes from the community were placed in the same community again (1.0 = always together). Values well above about 0.5 indicate a reproducible cluster. Low values indicate a community held together by a few patients.
""")

code(r"""
order = [g for c in louvain for g in sorted(c, key=lambda g: -freq[g])]
Zdf = pd.DataFrame(Z, index=net_genes, columns=net_genes)
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

sns.heatmap(Zdf.loc[order, order].clip(-6, 6), cmap="RdBu_r", center=0, ax=axes[0],
            cbar_kws={"label": "co-mutation z (red = together, blue = exclusive)"}, xticklabels=True, yticklabels=True)
axes[0].set_title("Null-adjusted co-mutation z, genes ordered by Louvain community")
sns.heatmap(consensus.loc[order, order], cmap="viridis", vmin=0, vmax=1, ax=axes[1],
            cbar_kws={"label": "fraction of bootstraps in same community"}, xticklabels=True, yticklabels=True)
axes[1].set_title(f"Consensus co-assignment across {N_BOOT} patient bootstraps")
for ax in axes:
    ax.tick_params(labelsize=7)
    edges = np.cumsum([len(c) for c in louvain])[:-1]
    for e in edges:
        ax.axhline(e, color="k", lw=0.6); ax.axvline(e, color="k", lw=0.6)
plt.tight_layout()
plt.show()
""")

code(r"""
pos = nx.spring_layout(H, weight="weight", seed=SEED, k=0.6)
palette = sns.color_palette("tab20", len(louvain))
fig, ax = plt.subplots(figsize=(13, 10))
fdr_edges = set(top_co[["Gene_A", "Gene_B"]].apply(frozenset, axis=1))
is_fdr = [frozenset(e) in fdr_edges for e in H.edges]
w = np.array([d["weight"] for _, _, d in H.edges(data=True)])
nx.draw_networkx_edges(H, pos, edgelist=[e for e, f in zip(H.edges, is_fdr) if not f],
                       width=(0.4 + 2.0 * w / w.max())[~np.array(is_fdr)], alpha=0.3, ax=ax)
nx.draw_networkx_edges(H, pos, edgelist=[e for e, f in zip(H.edges, is_fdr) if f],
                       width=3.5, edge_color="#C44E52", alpha=0.85, ax=ax)
nx.draw_networkx_nodes(H, pos, node_size=[30 + 3 * freq[g] for g in H],
                       node_color=[palette[community_of[g] - 1] for g in H], edgecolors="k", linewidths=0.4, ax=ax)
nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
ax.set_title(f"Gene co-mutation network (grey: p < {EDGE_P}; red: q < {FDR}; node size = frequency; colour = Louvain community)")
ax.axis("off")
plt.show()
""")

exec(open(__file__.rsplit("/",1)[0] + "/diagrams.py").read())
for _t, _s in DIAGRAM_CELLS:
    (md if _t == "markdown" else code)(_s)

md(r"""
## 5. Do the Clusters Map onto Clinical Subtypes?

Each patient is assigned to the gene community in which they carry the most mutations (social analogy: the group a user engages with most). Patients with no mutations in any connected gene are labelled *none*. The assignments are cross-tabulated against hormone-receptor status, histology and sample type. The heatmap shows **standardized Pearson residuals**: positive (red) means the community is over-represented in that clinical group.

Caution: these associations also explain some of the co-mutation edges. For example, if two genes are both enriched in lobular tumors, they will co-occur in a pooled cohort without interacting biologically. The limitations section returns to this.
""")
code(r"""
comm_counts = pd.DataFrame({i: X_net[[g for g in c if g in X_net]].sum(axis=1) for i, c in enumerate(louvain, start=1)})
patient_comm = comm_counts.idxmax(axis=1).astype(str).where(comm_counts.max(axis=1) > 0, "none")
patient_comm = "C" + patient_comm.where(patient_comm != "none", "-none")

clin = samples[["PATIENT_ID", "HR_STATUS", "CANCER_TYPE_DETAILED", "SAMPLE_TYPE"]].copy()
clin["Histology"] = clin["CANCER_TYPE_DETAILED"].map(
    lambda s: "Lobular" if "Lobular" in s and "Mixed" not in s else
              "Ductal" if "Ductal" in s and "Mixed" not in s else
              "Mixed" if "Mixed" in s else "Other")
clin["Community"] = patient_comm

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
assoc = []
for ax, var in zip(axes, ["HR_STATUS", "Histology", "SAMPLE_TYPE"]):
    sub = clin[~clin[var].isin(["Unk/ND"])]
    ct = pd.crosstab(sub["Community"], sub[var])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    resid = (ct - expected) / np.sqrt(expected)
    sns.heatmap(resid, annot=ct, fmt="d", cmap="RdBu_r", center=0, vmin=-6, vmax=6, ax=ax, cbar=ax is axes[-1])
    ax.set_title(f"{var}  (χ² p = {p:.1e})")
    assoc.append({"Variable": var, "chi2": round(chi2, 1), "dof": dof, "p": p})
plt.suptitle("Patient community vs clinical variables (cell = patient count, colour = standardized residual)")
plt.tight_layout()
plt.show()
pd.DataFrame(assoc)
""")

md(r"""
## 6. The Recommendation Engine

**Task:** a patient has a set of known mutated genes. Rank all other candidate genes by how likely the patient is to *also* carry a mutation in them. This is the "people who bought X also bought Y" problem. A practical use is prioritizing which genes to look at more closely (orthogonal validation, deeper coverage, germline follow-up) in a new tumor.

Six recommenders, from simplest to most "social":

| Model | Idea |
|---|---|
| **Popularity** | Recommend the most commonly mutated genes. The baseline every other model must beat. |
| **Item-kNN (cosine)** | Amazon-style item-to-item collaborative filtering on gene co-occurrence vectors. |
| **User-kNN (Jaccard, k=50)** | Find the 50 patients whose mutation profiles are most similar and pool their genes. |
| **Personalized PageRank** | Twitter "Who-To-Follow"-style random walk with restart on the patient–gene bipartite graph, starting at the patient's genes. |
| **NMF (latent programs)** | Matrix factorization. Patients are mixtures of latent mutational "programs". |
| **Null-adjusted co-mutation** | Base rate × the curveball-adjusted enrichment of each candidate with each known gene (naïve-Bayes style; uses this notebook's network). |

**Evaluation:** 5-fold cross-validation *over patients*. Each model is fitted (including its own curveball null) on 80% of patients. For every held-out patient with ≥2 mutated candidate genes, one mutated gene is hidden at random, the rest are given to the model, and the rank of the hidden gene is recorded. Metrics: **Hit@3**, **Hit@10** and **MRR** (mean reciprocal rank).

PIK3CA and TP53 together cover most patients, so popularity alone already scores well. The models are therefore also scored on the harder subset where the hidden gene is *not* PIK3CA or TP53. That subset is where co-mutation structure has to do the work.
""")
code(r"""
class Popularity:
    def fit(self, T):
        self.score_ = T.mean(axis=0)
        return self
    def score(self, q):
        return np.tile(self.score_, (len(q), 1))


class ItemKNN:
    def fit(self, T):
        C = T.T @ T
        norm = np.sqrt(np.diag(C))
        S = C / np.outer(norm, norm).clip(1e-12)
        np.fill_diagonal(S, 0)
        self.S_ = S
        self.pop_ = T.mean(axis=0)
        return self
    def score(self, q):
        return q @ self.S_ + 1e-6 * self.pop_   # popularity only breaks ties


class UserKNN:
    def __init__(self, k=50):
        self.k = k
    def fit(self, T):
        self.T_ = T
        self.pop_ = T.mean(axis=0)
        return self
    def score(self, q):
        inter = q @ self.T_.T
        union = q.sum(1)[:, None] + self.T_.sum(1)[None, :] - inter
        sim = inter / union.clip(1)
        out = np.zeros((len(q), self.T_.shape[1]))
        for r in range(len(q)):
            nn = np.argpartition(-sim[r], self.k)[:self.k]
            out[r] = sim[r, nn] @ self.T_[nn]
        return out + 1e-6 * self.pop_


class PersonalizedPageRank:
    # Random walk with restart on the bipartite graph; patients are down-weighted by burden
    # so hypermutated tumors don't act as hubs.
    def __init__(self, alpha=0.5, n_iter=30):
        self.alpha, self.n_iter = alpha, n_iter
    def fit(self, T):
        T = T.astype(float)
        P_gp = T.T / T.sum(axis=1).clip(1)                   # gene -> patient, weighted 1/burden
        P_gp = P_gp / P_gp.sum(axis=1, keepdims=True).clip(1e-12)
        P_pg = T / T.sum(axis=1, keepdims=True).clip(1)      # patient -> gene
        self.M_ = P_gp @ P_pg                                # two-step gene -> gene transition
        self.pop_ = T.mean(axis=0)
        return self
    def score(self, q):
        r0 = q / q.sum(1, keepdims=True).clip(1)
        r = r0.copy()
        for _ in range(self.n_iter):
            r = self.alpha * r0 + (1 - self.alpha) * r @ self.M_
        return r + 1e-6 * self.pop_


class NMFRecommender:
    def __init__(self, n_components=12):
        self.n_components = n_components
    def fit(self, T):
        self.model_ = NMF(self.n_components, init="nndsvda", max_iter=500, random_state=SEED).fit(T)
        self.pop_ = T.mean(axis=0)
        return self
    def score(self, q):
        return self.model_.transform(q) @ self.model_.components_ + 1e-6 * self.pop_


class NullAdjustedCoMutation:
    # score_j = log P(j) + sum_{i in profile} log((obs_ij + a) / (null_ij + a)), shrunk by pseudo-count a
    def __init__(self, n_null=N_NULL_BOOT, pseudo=2.0):
        self.n_null, self.pseudo = n_null, pseudo
    def fit(self, T):
        keep = T.sum(axis=0) > 0
        obs, null_mean, *_ = comutation_stats(pd.DataFrame(T[:, keep]), self.n_null)
        L = np.zeros((T.shape[1], T.shape[1]))
        L[np.ix_(keep, keep)] = np.log((obs + self.pseudo) / (null_mean + self.pseudo))
        np.fill_diagonal(L, 0)
        self.L_ = L
        self.logpop_ = np.log(T.mean(axis=0) + 1e-4)
        return self
    def score(self, q):
        return self.logpop_ + q @ self.L_


MODELS = {
    "Popularity": Popularity,
    "Item-kNN (cosine)": ItemKNN,
    "User-kNN (Jaccard, k=50)": UserKNN,
    "Personalized PageRank": PersonalizedPageRank,
    "NMF (12 programs)": NMFRecommender,
    "Null-adjusted co-mutation": NullAdjustedCoMutation,
}
""")

code(r"""
rec_genes = freq[freq >= MIN_PATIENTS_RECSYS].index.tolist()
R = X_full[rec_genes].values.astype(float)
print(f"Candidate genes for recommendation: {len(rec_genes)}")

eval_rng = np.random.default_rng(SEED)
records = []
for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(R)):
    T, E = R[tr], R[te]
    E = E[E.sum(axis=1) >= 2]
    hidden = np.array([eval_rng.choice(np.flatnonzero(row)) for row in E])
    Q = E.copy()
    Q[np.arange(len(Q)), hidden] = 0
    for name, cls in MODELS.items():
        S = cls().fit(T).score(Q)
        S = np.where(Q > 0, -np.inf, S)                      # never recommend a known gene
        target = S[np.arange(len(S)), hidden]
        rank = 1 + (S > target[:, None]).sum(axis=1)
        for h, rk in zip(hidden, rank):
            records.append({"Model": name, "fold": fold, "hidden_gene": rec_genes[h], "rank": rk})
    print(f"fold {fold + 1}/5 done: {len(E)} test patients")

ev = pd.DataFrame(records)
ev["hard"] = ~ev["hidden_gene"].isin(["PIK3CA", "TP53"])

def metrics(d):
    return pd.Series({"Hit@3": (d["rank"] <= 3).mean(), "Hit@10": (d["rank"] <= 10).mean(),
                      "MRR": (1 / d["rank"]).mean(), "n": len(d)})

res_all = ev.groupby("Model").apply(metrics, include_groups=False)
res_hard = ev[ev.hard].groupby("Model").apply(metrics, include_groups=False)
results_table = pd.concat({"All hidden genes": res_all, "Hidden gene not PIK3CA/TP53": res_hard}, axis=1)
results_table = results_table.loc[list(MODELS)].round(3)
results_table
""")

code(r"""
fold_mrr = ev[ev.hard].groupby(["Model", "fold"]).apply(lambda d: (1 / d["rank"]).mean(), include_groups=False).unstack()
fold_mrr = fold_mrr.loc[list(MODELS)]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
results_table.xs("Hit@10", axis=1, level=1).plot.barh(ax=axes[0])
axes[0].set_xlabel("Hit@10"); axes[0].set_title("Hidden gene ranked in top 10"); axes[0].invert_yaxis()
axes[0].legend(loc="lower right", fontsize=8)
fold_mrr.T.boxplot(ax=axes[1], vert=False)
axes[1].set_xlabel("MRR per fold (hard subset)"); axes[1].set_title("Fold-to-fold spread, hidden gene not PIK3CA/TP53")
axes[1].invert_yaxis()
plt.tight_layout()
plt.show()

# Paired comparison vs popularity on the hard subset (same patients, same hidden gene)
ev_hard = ev[ev.hard].copy()
ev_hard["pid"] = ev_hard.groupby("Model").cumcount()
paired = ev_hard.assign(rr=1 / ev_hard["rank"]).pivot(index="pid", columns="Model", values="rr")
# Paired bootstrap over test cases: reciprocal-rank gains are skewed and tie-heavy, so a CI on the mean is more honest than a rank test
boot_rng = np.random.default_rng(SEED)
boot_idx = boot_rng.integers(0, len(paired), size=(2000, len(paired)))
rows = []
for m in MODELS:
    if m == "Popularity":
        continue
    d = (paired[m] - paired["Popularity"]).values
    bm = d[boot_idx].mean(axis=1)
    rows.append({"Model": m, "Mean RR gain vs popularity": d.mean(), "95% CI low": np.percentile(bm, 2.5),
                 "95% CI high": np.percentile(bm, 97.5), "Cases improved": (d > 0).mean(), "Cases worsened": (d < 0).mean()})
pd.DataFrame(rows).round(4)
""")

md(r"""
### Which genes are recommendable?

Per-gene MRR for the best co-mutation model, compared with popularity, shows *which* genes the network structure helps predict. These are the genes whose mutation status is most informative about each other.
""")
code(r"""
per_gene = (ev.assign(rr=1 / ev["rank"]).pivot_table(index="hidden_gene", columns="Model", values="rr", aggfunc="mean"))
per_gene["n_hidden"] = ev[ev.Model == "Popularity"].groupby("hidden_gene").size()
best_model = results_table[("Hidden gene not PIK3CA/TP53", "MRR")].drop("Popularity").idxmax()
per_gene["Gain"] = per_gene[best_model] - per_gene["Popularity"]
print(f"Best non-baseline model on the hard subset: {best_model}")
per_gene[per_gene.n_hidden >= 10].sort_values("Gain", ascending=False)[["n_hidden", "Popularity", best_model, "Gain"]].round(3).head(15)
""")

md(r"""
## 7. Using the Engine on a Patient Profile

Edit `query_profile` below to any set of genes and re-run. The engine is refit on the **full** cohort. Each recommendation comes with its evidence: how many patients carrying each known gene also carry the candidate, and the curveball z for that pair. A clinician or researcher can then see *why* a gene was recommended rather than trusting a black-box score.
""")
code(r"""
final_models = {name: cls().fit(R) for name, cls in MODELS.items()}
Zfull = dict(zip(zip(pairs.Gene_A, pairs.Gene_B), pairs.z))
Zfull.update({(b, a): z for (a, b), z in list(Zfull.items())})


def recommend(profile, model=None, top_k=10):
    model = model or best_model
    unknown = [g for g in profile if g not in rec_genes]
    if unknown:
        print(f"Ignored (not a candidate gene): {unknown}")
    q = np.zeros((1, len(rec_genes)))
    q[0, [rec_genes.index(g) for g in profile if g in rec_genes]] = 1
    s = final_models[model].score(q)[0]
    s[q[0] > 0] = -np.inf
    top = np.argsort(-s)[:top_k]
    out = []
    for j in top:
        g = rec_genes[j]
        row = {"Recommended gene": g, "Score": round(float(s[j]), 3), "Cohort freq": f"{freq[g] / len(X_full):.1%}"}
        for k in profile:
            if k in X_full:
                both = int((X_full[k] & X_full[g]).sum())
                z = Zfull.get((k, g))
                row[f"w/ {k}"] = f"{both}/{int(X_full[k].sum())}" + (f" (z={z:+.1f})" if z is not None else "")
        out.append(row)
    return pd.DataFrame(out)


query_profile = ["CDH1"]
print(f"Profile {query_profile} — model: {best_model}")
recommend(query_profile)
""")

code(r"""
for profile in (["TP53"], ["PIK3CA", "MAP3K1"], ["ESR1"]):
    print(f"\nProfile {profile} — model: {best_model}")
    display(recommend(profile, top_k=8))
""")

md("## 8. Save Results")
code(r"""
pairs.to_csv("results/social_comutation_pairs.csv", index=False)
communities_df.to_csv("results/social_gene_communities.csv", index=False)
results_table.to_csv("results/social_recommender_benchmark.csv")
print("Saved: results/social_comutation_pairs.csv, results/social_gene_communities.csv, results/social_recommender_benchmark.csv")
""")

md(r"""
## 9. Findings

Numbers below are from the executed run (SEED = 677; every random step is seeded, so a re-run reproduces them).

**1. The null model changes the answer.** Only 3 of the top-20 pairs by raw co-occurrence count are also in the top 20 by curveball z. Raw counts mostly rediscover that PIK3CA and TP53 are common. After adjusting for burden and frequency, 11 co-occurring and 11 mutually exclusive pairs pass q < 0.10 (of 3,003 tested among the 78 genes mutated in ≥1% of patients). The moment-matched p-values agree with the empirical curveball p-values (Spearman ρ ≈ 0.999), so the FDR calls are not an artifact of the approximation.

**2. Mutual exclusivity is the strongest signal, and it is biologically coherent.**
- *Same-pathway redundancy* (one hit in the PI3K pathway is enough): PIK3CA × AKT1 (7 observed vs 37.7 expected), PIK3CA × PIK3R1 (1 vs 14.6), PIK3CA × PTEN (27 vs 49.7).
- *Subtype separation*: TP53 against the luminal genes GATA3, CDH1, MAP3K1, CBFB and ESR1. TP53 × ESR1 (17 vs 50.4) also reflects that ESR1 mutations arise under endocrine therapy, which HR-negative TP53-mutant tumors don't receive.

**3. Co-occurring clusters.** FDR-significant co-occurrences include several recognized breast-cancer associations:
- **CDH1 × ERBB2** (35 vs 12.9 expected, the strongest well-powered edge): activating ERBB2 *mutations* are known to be enriched in lobular carcinoma.
- **CDH1 × TBX3** and **PIK3CA × MAP3K1**: lobular / luminal-A co-mutation patterns.
- **GATA3 × CBFB** and **GATA3 × MEN1**: luminal transcription-factor programs.
- **TP53 × RB1** and **TP53 × NF1**: the HR-negative / basal-like axis.
- **EP300 × STK11**, **TET2 × SETD2** and **RICTOR × BRCA1** pass FDR on only 4 patients each. Treat them as leads. EP300 × STK11 overlaps the RB1-convergence cluster flagged in the earlier notebooks.

**4. Only some communities are reproducible.** Louvain and greedy modularity found the same 9 communities (ARI = 1.0, modularity 0.72). Label propagation broadly agrees (NMI 0.82). Bootstrap co-assignment separates real clusters from noise:
- **Stable:** GATA3 / CBFB / MEN1 (0.82), PTEN / SF3B1 (0.68), and the luminal-lobular cluster PIK3CA / CDH1 / MAP3K1 / TBX3 / FOXA1 / RUNX1 / ERBB2 / CDKN1B (0.60).
- **Not stable (0.17–0.35):** the TP53 hub community and the two rare-gene communities. The 11-gene community of large genes (SPEN, MLL2, NOTCH2, FLT4, PTPRT, APC …) is enriched in metastases and is most plausibly residual burden or gene-length structure rather than a biological program.

**5. The communities are subtype signatures.** Patient community membership is strongly associated with HR status (χ² p ≈ 1e-46) and histology (p ≈ 1e-41):
- The TP53 community is over-represented in HR-negative tumors.
- The PIK3CA/CDH1 community is over-represented in lobular and HR+ tumors.
- GATA3/CBFB/MEN1 is HR+ ductal.

This both validates the clusters and warns that pooled-cohort co-occurrence partly reflects subtype mix (Limitation 1).

**6. The recommender gives a modest, honest gain over popularity.** Over all hidden genes, popularity is hard to beat (MRR 0.266). The null-adjusted co-mutation model (0.280) and personalized PageRank (0.275) edge past it, and item-kNN and NMF fall well short. On the hard subset (hidden gene is not PIK3CA/TP53), the null-adjusted model has the best MRR (0.118 vs 0.102). A paired bootstrap puts its mean reciprocal-rank gain over popularity at +0.016 (95% CI 0.008–0.025). User-kNN is the only other model whose CI excludes zero (+0.012, CI 0.003–0.022). Item-kNN and personalized PageRank are indistinguishable from popularity, and NMF is worse. The per-gene table shows *where* the structure helps: **NF1, CBFB, ERBB2, FAT1, TBX3, FOXA1 and NCOR1** become 2–3× easier to rank once a patient's other mutations are known. These are the same genes that carry the significant network edges. Most genes, however, are not predictable from co-mutation alone. That is itself a finding: beyond subtype-level structure, most driver mutations in this panel are close to independent.
""")

md(r"""
## 10. Limitations

1. **Subtype confounding.** The cohort pools HR+/HR−, ductal/lobular, primary/metastatic tumors. Genes enriched in the same subtype will look co-mutated even without a biological interaction (Section 5 shows the clusters are strongly subtype-linked). The fix is a stratified null: run the curveball separately within each subtype and pool the statistics. That is a natural next step.
2. **Recommender scale.** The gains in Section 6 are modest. The engine is useful for ranking and for explaining co-mutation evidence, not for predicting an individual patient's genotype.
3. **Panel coverage proxy.** The "seen mutated on IMPACT341" rule for common genes is conservative but not exact. A gene on IMPACT341 that happened never to be mutated in those 432 samples would be wrongly dropped. Rare genes are the ones affected, and they are below the network frequency threshold anyway.
4. **Metastatic samples carry treatment-selected mutations** (for example, ESR1 after aromatase inhibitors). Their co-mutations partly reflect therapy history rather than tumor evolution.
5. **Moment-matched p-values** are an approximation to the curveball tail. They match the empirical p wherever that is resolvable, but the most extreme q-values (for example 1e-20) should be read as "far beyond the null" rather than literally.
6. **The recommender predicts co-occurrence, not causality.** A high-ranked gene is "frequently mutated in tumors like this one", not "this mutation will appear next".
7. **Mutation-level detail is collapsed to gene level.** For example, PIK3CA H1047R and E545K behave differently. `INCLUDE_CNA = True` adds amplifications and deep deletions as extra features, which matters for ERBB2, CCND1 and MYC in breast cancer.

## 11. References

- Strona G. et al. *A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals.* Nat. Commun. 5, 4114 (2014). (Curveball algorithm)
- Canisius S., Martens J.W.M., Wessels L.F.A. *A novel independence test for somatic alterations in cancer shows that biology drives mutual exclusivity but chance explains most co-occurrence.* Genome Biol. 17, 261 (2016). (DISCOVER)
- Blondel V.D. et al. *Fast unfolding of communities in large networks.* J. Stat. Mech. P10008 (2008). (Louvain)
- Raghavan U.N., Albert R., Kumara S. *Near linear time algorithm to detect community structures in large-scale networks.* Phys. Rev. E 76, 036106 (2007). (Label propagation)
- Linden G., Smith B., York J. *Amazon.com recommendations: item-to-item collaborative filtering.* IEEE Internet Computing 7(1) (2003).
- Gupta P. et al. *WTF: The Who to Follow service at Twitter.* WWW (2013). (Personalized PageRank recommendation)
- Liben-Nowell D., Kleinberg J. *The link-prediction problem for social networks.* JASIST 58(7) (2007). (Jaccard / Adamic–Adar)
- Razavi P. et al. *The genomic landscape of endocrine-resistant advanced breast cancers.* Cancer Cell 34(3) (2018). (Source cohort, breast_msk_2018)
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
import sys
nbf.write(nb, sys.argv[1])

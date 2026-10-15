DIAGRAM_CELLS = []
_dmd = lambda s: DIAGRAM_CELLS.append(("markdown", s.strip()))
_dcode = lambda s: DIAGRAM_CELLS.append(("code", s.strip()))

_dmd(r"""
### Cluster diagrams

The heatmaps and network above show every gene at once. The four diagrams below focus on the clusters themselves:

1. **Cluster map:** all clustered genes on a circle, grouped by community. It shows co-mutation inside clusters *and* mutual exclusivity between them.
2. **Cluster-to-cluster relationships:** average co-mutation z between every pair of communities.
3. **Inside each stable cluster:** the edges holding each reproducible cluster together, labelled observed / expected patients.
4. **Oncoprints:** the actual patients in each stable cluster, one column per patient.

Only clusters that hold up under bootstrap (co-assignment ≥ `STABLE_MIN`) get a colour. The rest are drawn in grey so they don't compete for attention. The three colours are a colorblind-safe set. Line *style*, not colour, separates co-mutation (solid) from exclusivity (dashed).
""")

_dcode(r"""
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STABLE_MIN = 0.5
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, validated colorblind-safe as a set
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#c3c2b7", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", SURFACE, "#e34948"])

stable = communities_df[communities_df["Bootstrap co-assignment"] >= STABLE_MIN]
stable_ids = stable["Community"].tolist()
if len(stable_ids) > len(SERIES):
    print(f"{len(stable_ids)} stable clusters; colouring the {len(SERIES)} most reproducible, the rest stay grey")
    stable_ids = stable.nlargest(len(SERIES), "Bootstrap co-assignment")["Community"].sort_values().tolist()
cluster_color = {cid: SERIES[i] for i, cid in enumerate(stable_ids)}


def cluster_genes(cid):
    return sorted(louvain[cid - 1], key=lambda g: -freq[g])


def cluster_name(cid, n=3):
    g = cluster_genes(cid)
    return f"C{cid} " + "/".join(g[:n]) + ("…" if len(g) > n else "")


def gene_color(g):
    return cluster_color.get(community_of.get(g), MUTED)


print("Stable clusters (coloured):")
for cid in stable_ids:
    print(f"  {cluster_name(cid, 10)}  — bootstrap co-assignment "
          f"{communities_df.set_index('Community').loc[cid, 'Bootstrap co-assignment']:.2f}")
""")

_dmd(r"""
#### 1. Cluster map

Genes sit on the circle grouped by Louvain community. A coloured arc marks each stable cluster. Genes that are in no cluster but take part in a significant exclusivity pair (ESR1, for example) are added as an *unclustered* group so their relationships stay visible.

- **Solid lines:** co-mutation edges from the clustering network (p < `EDGE_P`). Thick black lines pass FDR.
- **Dashed lines:** FDR-significant mutual exclusivity.
""")

_dcode(r"""
def bezier(p0, p1, pull=0.25, n=40):
    # Quadratic curve bowed toward the centre, as in a chord diagram
    c = (p0 + p1) / 2 * pull
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


groups = [cluster_genes(i + 1) for i in range(len(louvain))]
excl_genes = set(top_ex["Gene_A"]) | set(top_ex["Gene_B"])
unclustered = sorted(excl_genes - set(community_of), key=lambda g: -freq[g])
if unclustered:
    groups.append(unclustered)

gap = 1.5                                              # empty slots between groups
slots, ring = [], []
cursor = 0.0
for gi, grp in enumerate(groups):
    ring.append((cursor, cursor + len(grp) - 1, gi))
    slots += [(g, cursor + k) for k, g in enumerate(grp)]
    cursor += len(grp) + gap
total = cursor
angle = {g: 2 * np.pi * s / total for g, s in slots}
xy = {g: np.array([np.cos(a), np.sin(a)]) for g, a in angle.items()}

fig, ax = plt.subplots(figsize=(13, 13))
ax.set_aspect("equal"); ax.axis("off")

# Co-mutation edges (solid); FDR-significant ones bold
fdr_co = set(top_co[["Gene_A", "Gene_B"]].apply(frozenset, axis=1))
for a, b, d in H.edges(data=True):
    if a in xy and b in xy:
        sig = frozenset((a, b)) in fdr_co
        pts = bezier(xy[a], xy[b])
        ax.plot(*pts.T, color=INK if sig else INK_2, lw=2.6 if sig else 0.5 + d["weight"] / 6,
                alpha=0.9 if sig else 0.35, solid_capstyle="round", zorder=2)

# Mutual exclusivity edges (dashed)
for a, b, z in top_ex[["Gene_A", "Gene_B", "z"]].itertuples(index=False):
    if a in xy and b in xy:
        pts = bezier(xy[a], xy[b])
        ax.plot(*pts.T, color=INK_2, lw=0.6 + abs(z) / 4, ls=(0, (4, 3)), alpha=0.8, zorder=1)

# Cluster arcs
for start, end, gi in ring:
    cid = gi + 1
    if cid in cluster_color:
        th = np.linspace(2 * np.pi * (start - 0.4) / total, 2 * np.pi * (end + 0.4) / total, 60)
        ax.plot(1.07 * np.cos(th), 1.07 * np.sin(th), color=cluster_color[cid], lw=7, solid_capstyle="round")
        mid = (th[0] + th[-1]) / 2
        ax.text(1.42 * np.cos(mid), 1.42 * np.sin(mid), f"C{cid}", color=INK, fontsize=12, fontweight="bold",
                ha="center", va="center")

# Gene nodes (2px surface ring) and radial labels
for g, p in xy.items():
    ax.scatter(*p, s=40 + 2.2 * freq[g], color=gene_color(g), edgecolor="white", linewidth=1.5, zorder=3)
    a = np.degrees(angle[g])
    flip = 90 < a < 270
    ax.text(*(1.13 * p), g, rotation=a + 180 if flip else a, rotation_mode="anchor",
            ha="right" if flip else "left", va="center", fontsize=9, color=INK)

legend = [Patch(color=cluster_color[c], label=cluster_name(c, 4)) for c in stable_ids]
legend += [Patch(color=MUTED, label="unstable cluster / unclustered"),
           Line2D([], [], color=INK, lw=2.6, label=f"co-mutation, q < {FDR}"),
           Line2D([], [], color=INK_2, lw=1, alpha=0.6, label=f"co-mutation, p < {EDGE_P}"),
           Line2D([], [], color=INK_2, lw=1.5, ls=(0, (4, 3)), label=f"mutual exclusivity, q < {FDR}")]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=False, fontsize=10)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
ax.set_title("Gene cluster map: what mutates together, and what avoids each other", fontsize=14, color=INK)
plt.show()
""")

_dmd(r"""
#### 2. How the clusters relate to each other

Each cell is the mean null-adjusted z over all gene pairs between two communities. The diagonal is the cluster's internal cohesion. Red means the clusters' genes tend to mutate together; blue means they tend to avoid each other. Numbers are printed in every cell, so colour is never the only cue.
""")

_dcode(r"""
cids = list(range(1, len(louvain) + 1))
between = pd.DataFrame(index=cids, columns=cids, dtype=float)
for a in cids:
    for b in cids:
        ga, gb = cluster_genes(a), cluster_genes(b)
        block = Zdf.loc[ga, gb].values
        if a == b:
            block = block[np.triu_indices(len(ga), k=1)] if len(ga) > 1 else np.array([np.nan])
        between.loc[a, b] = np.nanmean(block)
labels_cc = [cluster_name(c) + (" ●" if c in cluster_color else "") for c in cids]
lim = np.nanmax(np.abs(between.values))

fig, ax = plt.subplots(figsize=(11, 8.5))
sns.heatmap(between.astype(float), cmap=DIVERGING, center=0, vmin=-lim, vmax=lim, annot=True, fmt=".1f",
            linewidths=2, linecolor="white", square=True, xticklabels=[f"C{c}" for c in cids], yticklabels=labels_cc,
            cbar_kws={"label": "mean co-mutation z (red = together, blue = exclusive)", "shrink": 0.7}, ax=ax)
for t in ax.get_yticklabels():
    cid = int(t.get_text().split()[0][1:])
    if cid in cluster_color:
        t.set_fontweight("bold")
ax.set_title("Cluster-to-cluster co-mutation (● = bootstrap-stable cluster)", color=INK)
plt.tight_layout()
plt.show()
""")

_dmd(r"""
#### 3. Inside each stable cluster

Each stable cluster is drawn on its own with every co-mutation edge (p < `EDGE_P`) that holds it together. Edge labels show **observed / expected** co-mutated patients under the curveball null. Bold edges pass FDR. Node size is cohort frequency.
""")

_dcode(r"""
fig, axes = plt.subplots(1, len(stable_ids), figsize=(6.5 * len(stable_ids), 6.5), squeeze=False)
obs_df = pd.DataFrame(obs, index=net_genes, columns=net_genes)
exp_df = pd.DataFrame(exp_null, index=net_genes, columns=net_genes)
for ax, cid in zip(axes[0], stable_ids):
    genes_c = cluster_genes(cid)
    sub = H.subgraph(genes_c)
    p = nx.kamada_kawai_layout(sub) if sub.number_of_edges() else nx.circular_layout(sub)
    for a, b, d in sub.edges(data=True):
        sig = frozenset((a, b)) in fdr_co
        ax.plot(*zip(p[a], p[b]), color=INK if sig else INK_2, lw=3 if sig else 1.2, alpha=0.9 if sig else 0.5, zorder=1)
        mid = (p[a] + p[b]) / 2
        ax.text(*mid, f"{obs_df.loc[a, b]:.0f} / {exp_df.loc[a, b]:.1f}", fontsize=8.5, color=INK_2, ha="center",
                va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9), zorder=2)
    for g in genes_c:
        ax.scatter(*p[g], s=200 + 4 * freq[g], color=cluster_color[cid], edgecolor="white", linewidth=2, zorder=3)
        ax.text(p[g][0], p[g][1] - (0.08 + 0.006 * np.sqrt(freq[g])), f"{g}\n{freq[g] / len(X_full):.1%}", ha="center", va="top", fontsize=9.5,
                color=INK, zorder=4)
    score = communities_df.set_index("Community").loc[cid, "Bootstrap co-assignment"]
    ax.set_title(f"C{cid} — bootstrap co-assignment {score:.2f}", color=INK)
    ax.margins(0.25); ax.axis("off")
fig.suptitle("Stable gene clusters (edge label = observed / expected co-mutated patients; bold = q < FDR)", color=INK)
plt.tight_layout()
plt.show()
""")

_dmd(r"""
#### 4. Oncoprints: the patients behind each cluster

One column per patient carrying at least one mutation in the cluster, one row per gene. Columns are sorted so that patients sharing a pattern sit next to each other. Co-mutation shows up as columns filled in several rows at once; exclusivity shows up as a "staircase". The top track gives the patient's hormone-receptor status, since Section 5 shows the clusters are subtype-linked.
""")

_dcode(r"""
hr = samples.loc[X_full.index, "HR_STATUS"]
hr_code = {"Positive": 2, "Negative": 1}
heights = [len(cluster_genes(c)) + 1.2 for c in stable_ids]
fig, axes = plt.subplots(len(stable_ids), 1, figsize=(15, 0.42 * sum(heights) + 1.5),
                         gridspec_kw={"height_ratios": heights}, squeeze=False)
for ax, cid in zip(axes[:, 0], stable_ids):
    genes_c = cluster_genes(cid)
    M = X_full[genes_c]
    M = M[M.sum(axis=1) > 0]
    M = M.sort_values(genes_c, ascending=False)          # lexicographic: groups shared patterns
    track = hr.loc[M.index].map(hr_code).fillna(0).values[None, :]
    grid = np.vstack([track + 10, M.T.values])           # row 0 = HR track, then genes
    cmap = ListedColormap([SURFACE, cluster_color[cid]] + ["white"] * 8 + ["white", MUTED, INK_2])
    ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=12)
    ax.set_yticks(range(len(genes_c) + 1))
    ax.set_yticklabels(["HR status"] + genes_c, fontsize=9.5, color=INK)
    ax.set_xticks([])
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(range(1, len(genes_c) + 1))
    ax2.set_yticklabels([f"{M[g].sum()} ({M[g].mean():.0%})" for g in genes_c], fontsize=8.5, color=INK_2)
    ax2.grid(False)
    for s in list(ax.spines.values()) + list(ax2.spines.values()):
        s.set_visible(False)
    ax.grid(False)
    ax.axhline(0.5, color="white", lw=3)
    multi = (M.sum(axis=1) >= 2).mean()
    ax.set_title(f"{cluster_name(cid, 10)} — {len(M)} patients, {multi:.0%} carry ≥2 cluster genes",
                 loc="left", color=INK, fontsize=11)
fig.legend(handles=[Patch(color=INK_2, label="HR-positive"), Patch(color=MUTED, label="HR-negative"),
                    Patch(facecolor="white", edgecolor=MUTED, label="HR unknown (blank)"), Patch(color=SURFACE, label="not mutated")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
""")

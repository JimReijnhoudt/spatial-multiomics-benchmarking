import seaborn as sns
import spatialmeta as smt
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


smt.data.list_datasets()


joint_adata = smt.data.load_adata(
    sample_name="Y7_T_raw",
    modality="joint"
)

joint_adata.write_h5ad("SpatialMETA/data/Y7_T_joint.h5ad")


print(joint_adata)


joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata)


joint_adata.layers["counts"] = joint_adata.X.copy()


smt.pp.normalize_total_joint_adata_sm_st(joint_adata,
                         target_sum_SM=1e4,
                         target_sum_ST=1e4)


joint_adata.layers["normalized"] = joint_adata.X.copy()


joint_adata.raw = joint_adata


smt.pp.spatial_variable_joint_adata_sm_st(joint_adata,
                                         n_top_genes = 2000,
                                         n_top_metabolites = 800,
                                         add_key = "highly_variable_moranI")


joint_adata = joint_adata[:,joint_adata.var.highly_variable_moranI]


joint_adata.write_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800.h5ad")


joint_adata = sc.read_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800.h5ad")


joint_adata.X = joint_adata.layers["counts"]


smt.pp.normalize_total_joint_adata_sm_st(
    joint_adata,
    target_sum_SM=1e3,
    target_sum_ST=None
)


model = smt.model.ConditionalVAESTSM(
    joint_adata,
    device='cpu', # Small change to CPU instead of CUDA
    reconstruction_method_sm='g',
    reconstruction_method_st='zinb',
)


loss_dict = model.fit(
    max_epoch=64,
    lr=1e-3,
    mode='single'
)


fig,axes = plt.subplots(3, 3, figsize=(20,10))
axes = axes.flatten()

for ax, (k, v) in zip(axes, loss_dict.items()):
    ax.plot(v)
    ax.set_title(k)

plt.tight_layout()
plt.show()


Z = model.get_latent_embedding()
X = model.get_normalized_expression()
C = model.get_modality_contribution()


joint_adata.layers['reconstruction'] = X
joint_adata.obsm['X_emb']=Z
joint_adata.obs['contribution_st']=C
joint_adata.obs['contribution_sm']=1-C


sc.pp.neighbors(
    joint_adata,
    use_rep="X_emb",
    n_neighbors=15
)


sc.tl.umap(
    joint_adata,
    min_dist=1,
    spread=1
)


sc.tl.leiden(
    joint_adata,
    key_added="VAE_clusters_latent10"
)


sc.pl.umap(
    joint_adata,
    color=["VAE_clusters_latent10"],
    show=True
)


sc.pl.spatial(
    joint_adata,
    img_key="hires",
    color=["VAE_clusters_latent10"]
)


sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["CD8A","137.04580812911064"],
              layer="normalized")


sc.pl.spatial(joint_adata,
              img_key="hires",
              color_map = "vlag",
              color=["CD8A","137.04580812911064"],
              layer="reconstruction")


sc.pl.spatial(
    joint_adata,
    img_key="hires",
    color_map = smt.pl.make_colormap(['#2ec4b6','#ffffff','#ff9f1c' ]),
    color=['contribution_st','contribution_sm'],
    layer="normalized",
    alpha_img=0.1,
    size=1.5
)


obs_df = joint_adata.obs
obs_filter_df = pd.concat([
    obs_df[['VAE_clusters_latent10', 'contribution_st']].rename(columns={'contribution_st': 'contribution'}).assign(type='st'),
    obs_df[['VAE_clusters_latent10', 'contribution_sm']].rename(columns={'contribution_sm': 'contribution'}).assign(type='sm')
])
fig,ax = smt.pl.create_fig(
    figsize = (12,4)
)
sns.violinplot(
    data=obs_filter_df,
    x="VAE_clusters_latent10",
    y="contribution",
    hue="type",
    split=True,
    inner="quart",
    palette=['#2ec4b6', '#FFCC70'],
    scale='width',  # Make violins the same width
    bw=0.2,         # Adjust smoothness (lower value = fatter violins)
    cut=0           # Limit the violin to data range
)

plt.xticks(rotation=90)
plt.show()


joint_adata.write_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800_spatialmeta.h5ad")


from multi_benmark_function import (
    compute_CHAOS, compute_PAS,
    marker_score,
    compute_ARI, compute_NMI,
    compute_ASW, compute_gt_silhouette,
    compute_clisi_graph, compute_isolated_aws,
    calculate_gene_specificity,
    logistic_regression_feature_importance,
    calculate_mutual_information
)
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

PRED_KEY = 'VAE_clusters_latent10'
EMB_KEY   = 'X_emb'
GT_KEY    = 'pathological_annotation'


import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Pull out dense matrices
X_orig  = joint_adata.layers['counts']
X_recon = joint_adata.layers['reconstruction']

if hasattr(X_orig, 'toarray'):
    X_orig = X_orig.toarray()
if hasattr(X_recon, 'toarray'):
    X_recon = X_recon.toarray()

# PCC — one value per spot, then averaged
pccs = [pearsonr(X_orig[i], X_recon[i])[0] for i in range(X_orig.shape[0])]
mean_pcc = np.nanmean(pccs)

# Cosine Similarity — diagonal of the pairwise matrix gives per-spot values
cs_matrix = cosine_similarity(X_orig, X_recon)
mean_cs = np.nanmean(cs_matrix.diagonal())

print(f"Mean PCC: {mean_pcc:.4f}")
print(f"Mean Cosine Similarity: {mean_cs:.4f}")


chaos_score = compute_CHAOS(
    joint_adata,
    pred_key='VAE_clusters_latent10',
    spatial_key='spatial'         # adata.obsm['spatial']
)
print(f"CHAOS: {chaos_score:.4f}")


pas_score = compute_PAS(
    joint_adata,
    pred_key='VAE_clusters_latent10',
    spatial_key='spatial'
)
print(f"PAS: {pas_score:.4f}")


# Prep: set X to counts and make sure leiden key matches
adata_ms = joint_adata.copy()
adata_ms.X = adata_ms.layers['counts'].copy()
adata_ms.obs[PRED_KEY] = adata_ms.obs[PRED_KEY].astype('category')

# ST modality
adata_ST = adata_ms[:, adata_ms.var['type'] == 'ST'].copy()
moranI_ST, gearyC_ST = marker_score(adata_ST, domain_key=PRED_KEY, top_n=50)
print(f"ST  Moran's I: {moranI_ST:.4f}  |  Geary's C: {gearyC_ST:.4f}")

# SM modality
adata_SM = adata_ms[:, adata_ms.var['type'] == 'SM'].copy()
moranI_SM, gearyC_SM = marker_score(adata_SM, domain_key=PRED_KEY, top_n=50)
print(f"SM  Moran's I: {moranI_SM:.4f}  |  Geary's C: {gearyC_SM:.4f}")

# Moran's I: higher (→1) = stronger spatial structure
# Geary's C: lower (→0) = stronger spatial structure


specificity_SM = calculate_gene_specificity(adata_SM, domain_key=PRED_KEY, layers="counts")
specificity_ST = calculate_gene_specificity(adata_ST, domain_key=PRED_KEY, layers="counts")

print(f"SM  specificity: {specificity_SM:.4f}  |  ST specificity: {specificity_ST:.4f}")


logistic_ST = logistic_regression_feature_importance(adata_ST, domain_key=PRED_KEY, layers="counts")
logistic_SM = logistic_regression_feature_importance(adata_SM, domain_key=PRED_KEY, layers="counts")

print(f"SM  feature importance: {logistic_SM:.4f}  |  ST feature importance: {logistic_ST:.4f}")


mi_ST = calculate_mutual_information(adata_SM, domain_key=PRED_KEY, layers="counts")
mi_SM = calculate_mutual_information(adata_ST, domain_key=PRED_KEY, layers="counts")

print(f"SM  mutual information: {logistic_SM:.4f}  |  ST mutual information: {logistic_ST:.4f}")


ari = compute_ARI(joint_adata, gt_key=PRED_KEY, pred_key=PRED_KEY)
nmi = compute_NMI(joint_adata, gt_key=PRED_KEY, pred_key=PRED_KEY)
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")
# Both range 0–1, higher is better


# Spatial ASW — silhouette in physical space w.r.t. predicted clusters
asw_spatial = compute_ASW(
    joint_adata,
    pred_key=PRED_KEY,
    spatial_key='spatial'
)
print(f"Spatial ASW: {asw_spatial:.4f}")

# Embedding ASW — silhouette in X_emb space w.r.t. ground truth (scib version)
# Requires GT_KEY in adata.obs
asw_emb = compute_gt_silhouette(
    joint_adata,
    gt_key=PRED_KEY,
    embedding_key=EMB_KEY
)
print(f"Embedding ASW (gt): {asw_emb:.4f}")
# Range -1 to 1, higher is better


clisi = compute_clisi_graph(
    joint_adata,
    gt_key=PRED_KEY,
    embedding_key=EMB_KEY
)
print(f"cLISI: {clisi:.4f}")
# Higher = cell-type neighborhoods are purer in embedding space


isolated_asw = compute_isolated_aws(
    joint_adata,
    gt_key=PRED_KEY,
    embedding_key=EMB_KEY
)
print(f"Isolated label ASW: {isolated_asw:.4f}")
# Higher = rare cell types are well-separated in embedding space


import pandas as pd

results = {
    'PCC':           mean_pcc,
    'CosineSim':     mean_cs,
    'CHAOS':         chaos_score,
    'PAS':           pas_score,
    'ST_MoranI':     moranI_ST,
    'ST_GearyC':     gearyC_ST,
    'SM_MoranI':     moranI_SM,
    'SM_GearyC':     gearyC_SM,
    'ST_Specificity': specificity_ST,
    'SM_Specificity': specificity_SM,
    'ST_Logistic': logistic_ST,
    'SM_Logistic': logistic_SM,
    'ST_Mutual Information': mi_ST,
    'SM_Mutual Information': mi_SM,
    'ARI':           ari,
    'NMI':           nmi,
    'ASW_spatial':   asw_spatial,
    'ASW_emb':       asw_emb,
    'cLISI':         clisi,
    'Isolated_ASW':  isolated_asw,
}

reconstrucion_accuracy = (mean_pcc + mean_cs)/2
continuity_score = ((1 - chaos_score) + (1 - pas_score))/2
# MoransI + 1-Geary'sG + Specificity + Logistic + MI / 5
marker_score_ST = (moranI_ST + (1 - gearyC_ST) + specificity_ST + mi_ST)/5
marker_score_SM = (moranI_SM + (1 - gearyC_SM) + specificity_SM + mi_SM)/5
marker_score = (moranI_ST + (1 - gearyC_ST) + specificity_ST + mi_ST + moranI_SM + (1 - gearyC_SM) + specificity_SM + mi_SM) / 10
biological_conservation = (ari + nmi + asw_spatial + isolated_asw + clisi)/5

results_metrics = {"Continuity Score": continuity_score,
                   "Marker score": marker_score,
                   "Marker Score ST": marker_score_ST,
                   "Marker score SM": marker_score_SM,
                   "Biological Conservation": biological_conservation,
                   "Reconstruction Accuracy": reconstrucion_accuracy
                   }

results = pd.DataFrame(results, index=['spatialMETA'])

results_metrics = pd.DataFrame(results_metrics, index=['spatialMETA'])


results
results_metrics
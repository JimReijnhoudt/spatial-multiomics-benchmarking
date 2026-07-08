def run_vertical_spatialmeta(iteration):

    # # Vertical integration for spatialmeta
    # 
    # # Loading data & Preprocessing


    # ## Import statements


    # This tutorial demonstrates the process of vertically integrating spatial transcriptomics (ST) and spatial metabolomics (SM) data using spatialMETA.


    import seaborn as sns
    import spatialmeta as smt
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt


    # ## Downloading and storing data


    # In this section, we read and preprocess the joint SM and ST data. The steps include removing unwanted genes, normalizing the data, identifying spatial variable genes and metabolites, and selecting the subset of spatial variable genes and metaboliyes for further analysis.


    # Load the AnnData object for the given sample name and modality.
    # 
    # Parameters: <br>
    # **sample_name** – str The name of the sample. Use list_datasets to get the list of all available datasets. <br>
    # **modality** – Literal[“ST”, “SM”, “joint”] The modality of the dataset. Choose from “ST”, “SM”, or “joint”. <br>


    smt.data.list_datasets()


    joint_adata = smt.data.load_adata(
        sample_name="Y7_T_raw",
        modality="joint"
    )

    joint_adata.write_h5ad("SpatialMETA/data/Y7_T_joint.h5ad")


    print(joint_adata)


    # ## Preprocessing


    # removing unwanted genes (Not listed on spatialmeta docs). <br>
    # pp. stands for preprocessing
    # 


    joint_adata = smt.pp.removeHSP_MT_RPL_DNAJ(joint_adata)


    joint_adata.layers["counts"] = joint_adata.X.copy()


    # Normalize the total intensity of the SM and ST data in the joint AnnData object.
    # 
    # Parameters: <br>
    # **joint_adata** – AnnDataJointSMST. The joint AnnData object with SM and ST data. <br>
    # **target_sum_SM** – Optional[int]. The target sum for SM data, default is 1e4. <br>
    # **target_sum_ST** – Optional[int]. The target sum for ST data, default is 1e4. <br>


    smt.pp.normalize_total_joint_adata_sm_st(joint_adata,
                            target_sum_SM=1e4,
                            target_sum_ST=1e4)


    joint_adata.layers["normalized"] = joint_adata.X.copy()


    joint_adata.raw = joint_adata


    # Calculate the spatial variables for the joint AnnData object and remove the batch-specific spatial variables.
    # 
    # Parameters: <br>
    # joint_adata – AnnDataJointSMST. The joint AnnData object with SM and ST data. <br>
    # **n_top_genes** – int. The number of top genes, default is 2000. <br>
    # **n_top_metabolites** – int. The number of top metabolites, default is 800. <br>
    # **add_key** – str. The key for the spatial variables, default is “highly_variable_moranI”. <br>
    # batch_key – Optional[str]. The batch key, default is None. <br>
    # min_samples – int. The minimum number of samples, default is 2. <br>
    # min_frac – float. The minimum fraction, default is 0.8. <br>
    # min_logfc – float. The minimum log fold change, default is 3. <br>


    smt.pp.spatial_variable_joint_adata_sm_st(joint_adata,
                                            n_top_genes = 2000,
                                            n_top_metabolites = 800,
                                            add_key = "highly_variable_moranI")


    joint_adata = joint_adata[:,joint_adata.var.highly_variable_moranI]


    joint_adata.write_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800.h5ad")


    # # ConditionalVAE model for vertical integration


    # In this section, we train a ConditionalVAE model on the preprocessed joint SM and ST data. The VAE model learns a low-dimensional representation of the data that captures the underlying structure and relationships between the SM and ST measurements.


    joint_adata = sc.read_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800.h5ad")


    joint_adata.X = joint_adata.layers["counts"]


    # Normalize the total intensity of the SM and ST data in the joint AnnData object.
    # 
    # Parameters: <br>
    # **joint_adata** – AnnDataJointSMST. The joint AnnData object with SM and ST data. <br>
    # **target_sum_SM** – Optional[int]. The target sum for SM data, default is 1e4. <br>
    # **target_sum_ST** – Optional[int]. The target sum for ST data, default is 1e4. <br>


    smt.pp.normalize_total_joint_adata_sm_st(
        joint_adata,
        target_sum_SM=1e3,
        target_sum_ST=None
    )


    # This class implements a Conditional Variational Autoencoder with Mixture of Experts (MoE) for vertical and horizontal integration of ST and SM.
    # 
    # Parameters: <br>
    # adata – AnnDataJointSMST object containing the spatial multi-omics data. <br>
    # hidden_stacks – List of integers specifying the number of hidden units in each stack of the encoder and decoder, default is [128]. <br>
    # batch_keys – Optional list of strings specifying the batch keys for batch correction. <br>
    # n_latent – Integer specifying the dimensionality of the latent space, default is 10. <br>
    # bias – Boolean indicating whether to include bias terms in the linear layers, default is True. <br>
    # use_batch_norm – Boolean indicating whether to use batch normalization in the linear layers, default is True. <br>
    # use_layer_norm – Boolean indicating whether to use layer normalization in the linear layers, default is False. <br>
    # dropout_rate – Float specifying the dropout rate for the linear layers, default is 0.1. <br>
    # activation_fn – Callable specifying the activation function to use in the linear layers, default is nn.ReLU. <br>
    # **device** – String or torch.device specifying the device to use for computation, default is “cpu”. <br>
    # batch_embedding – Literal[“embedding”, “onehot”] specifying the type of batch embedding to use, default is “onehot”. <br>
    # encode_libsize – Boolean indicating whether to encode library size information, default is False. <br>
    # batch_hidden_dim – Integer specifying the dimensionality of the batch hidden layer, default is 8. <br>
    # **reconstruction_method_st** – Literal[‘mse’, ‘zg’, ‘zinb’] specifying the reconstruction method for the spatial data, default is ‘zinb’. <br>
    # **reconstruction_method_sm** – Literal[‘mse’, ‘zg’, ‘g’] specifying the reconstruction method for the single-cell multi-omics data, default is ‘g’. <br>


    model = smt.model.ConditionalVAESTSM(
        joint_adata,
        device='cpu', # Small change to CPU instead of CUDA
        reconstruction_method_sm='g',
        reconstruction_method_st='zinb',
    )


    # Fits the model.
    # 
    # Parameters: <br>
    # **max_epoch** – Integer specifying the maximum number of epochs to train the model, default is 35. <br>
    # n_per_batch – Integer specifying the number of samples per batch, default is 128. <br>
    # **mode** – Optional string specifying the mode of training. Can be either ‘single’ or ‘multi’, default is None. <br>
    # reconstruction_reduction – String specifying the reduction method for the reconstruction loss, default is ‘sum’. <br>
    # kl_weight – Float specifying the weight of the KL divergence loss, default is 1. <br>
    # reconstruction_st_weight – Float specifying the weight of the reconstruction loss for spatial transcriptomics, default is 1. <br>
    # reconstruction_sm_weight – Float specifying the weight of the reconstruction loss for single-cell multi-omics, default is 1. <br>
    # reconstruction_st_corr_weight – Float specifying the weight of the correlation reconstruction loss for spatial transcriptomics, default is 1. <br>
    # reconstruction_sm_corr_weight – Float specifying the weight of the correlation reconstruction loss for single-cell multi-omics, default is 1. <br>
    # n_epochs_kl_warmup – Integer specifying the number of epochs for KL divergence warmup, default is 400. <br>
    # optimizer_parameters – Iterable specifying the parameters for the optimizer, default is None. <br>
    # weight_decay – Float specifying the weight decay for the optimizer, default is 1e-6. <br>
    # **lr** – Float specifying the learning rate for the optimizer. <br>
    # random_seed – Integer specifying the random seed, default is 12. <br>
    # kl_loss_reduction – String specifying the reduction method for the KL divergence loss, default is ‘mean’. <br>
    # mmd_weight – Float specifying the weight of the MMD loss, default is 1. <br>
    # 
    # Returns: <br>
    # Dictionary containing the training loss values. <br>


    loss_dict = model.fit(
        max_epoch=64,
        lr=1e-3,
        mode='single'
    )


    # ## Loss visualisation


    # In this section, we visualize the loss during the training of the ConditionalVAE model. The loss consists of three components: Kullback-Leibler divergence loss, reconstruction loss for spatial metabolomics (SM), and reconstruction loss for spatial transcriptomics (ST).


    fig,axes = plt.subplots(3, 3, figsize=(20,10))
    axes = axes.flatten()

    for ax, (k, v) in zip(axes, loss_dict.items()):
        ax.plot(v)
        ax.set_title(k)

    plt.tight_layout()
    plt.savefig(f"benchmarks/results/automated/plots/loss_visualization_{iteration}.png")
    #plt.show()


    # ## Visualisation and analysis


    # In this section, we visualize the results of the vertical integration using various plots and analysis techniques. We plot the loss curves during model training, visualize the latent space using UMAP, and explore the spatial patterns of specific genes or metabolites after denoising.


    Z = model.get_latent_embedding()
    X = model.get_normalized_expression()
    C = model.get_modality_contribution()


    joint_adata.layers['reconstruction'] = X
    joint_adata.obsm['X_emb']=Z
    joint_adata.obs['contribution_st']=C
    joint_adata.obs['contribution_sm']=1-C


    # Compute the nearest neighbors distance matrix and a neighborhood graph of observations
    # 
    # Parameters: <br>
    # **adata** (AnnData) - Annotated data matrix. <br>
    # **use_rep** (str) - Use the indicated representation. 'X' or any key for .obsm is valid. <br>
    # **n_neighbors** (int) - The size of local neighborhood (in terms of number of neighboring data points) used for manifold approximation. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. <br>


    sc.pp.neighbors(
        joint_adata,
        use_rep="X_emb",
        n_neighbors=15
    )


    # Embed the neighborhood graph using UMAP. UMAP (Uniform Manifold Approximation and Projection) is a manifold learning technique suitable for visualizing high-dimensional data.
    # 
    # Parameters: <br>
    # **adata** (AnnData) - Annotated data matrix. <br>
    # **min_dist** (float) - The effective minimum distance between embedded points. Smaller values will result in a more clustered/clumped embedding where nearby points on the manifold are drawn closer together, while larger values will result on a more even dispersal of points. <br>
    # **spread** (float) - The effective scale of embedded points. In combination with min_dist this determines how clustered/clumped the embedded points are. <br>


    sc.tl.umap(
        joint_adata,
        min_dist=1,
        spread=1
    )


    # Cluster cells into subgroups. Cluster cells using the Leiden algorithm, an improved version of the Louvain algorithm.
    # 
    # Parameters: <br>
    # **adata** (AnnData) - Annotated data matrix. <br>
    # **key_added** (str) - (adata.obs) key under which to add the cluster labels. <br>
    # 


    sc.tl.leiden(
        joint_adata,
        key_added="VAE_clusters_latent10"
    )


    # ### Cluster plot


    # Scatter plot in UMAP basis.
    # https://scanpy.scverse.org/en/stable/api/generated/scanpy.pl.umap.html#scanpy.pl.umap


    sc.pl.umap(
        joint_adata,
        color=["VAE_clusters_latent10"],
        show=True
    )


    # ### Hires plots


    # Scatter plot in spatial coordinates.
    # https://scanpy.scverse.org/en/stable/api/generated/scanpy.pl.spatial.html#scanpy.pl.spatial


    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color=["VAE_clusters_latent10"]
    )


    # ### Normalized plots 1 (purple)


    sc.pl.spatial(joint_adata,
                img_key="hires",
                color_map = "vlag",
                color=["CD8A","137.04580812911064"],
                layer="normalized")


    # ### Recondstructed plots


    sc.pl.spatial(joint_adata,
                img_key="hires",
                color_map = "vlag",
                color=["CD8A","137.04580812911064"],
                layer="reconstruction")


    # ### Normalized plots 2 (cyan)


    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color_map = smt.pl.make_colormap(['#2ec4b6','#ffffff','#ff9f1c' ]),
        color=['contribution_st','contribution_sm'],
        layer="normalized",
        alpha_img=0.1,
        size=1.5
    )


    # ### Violin plots


    # Create a figure with the specified size and axis properties.
    # 
    # Parameters: <br>
    # **figsize** – tuple specifying the size of the figure, default is (8, 4). <br>
    # 
    # Returns: <br>
    # figure and axis objects. <br>
    # 
    # A violin plot plays a similar role as a box-and-whisker plot. It shows the distribution of data points after grouping by one (or more) variables. Unlike a box plot, each violin is drawn using a kernel density estimate of the underlying distribution. <br>
    # https://seaborn.pydata.org/generated/seaborn.violinplot.html


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
    plt.savefig(f"benchmarks/results/automated/plots/violin_plot_{iteration}.png")
    #plt.show()


    joint_adata.write_h5ad("SpatialMETA/data/Y7_T_adata_joint_hvf2800_spatialmeta.h5ad")


    # ## Benchmarking metrics
    # 
    # ### Setup


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


    # ### Reconstruction accuracy
    # 
    # PCC & cosine similarity


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


    # ### Continuity score
    # 
    # These metrics assess the smoothness and consistency of the data embedding. 
    # 
    # CHAOS — measures spatial compactness of clusters by averaging nearest-neighbor distances within each cluster. Lower is more compact/smooth.
    # 
    # PAS (Proportion of Ambiguous Spots) — for each spot, checks whether the majority of its 10 nearest spatial neighbors belong to a different cluster. Returns the fraction of such "ambiguous" spots. Lower is more spatially consistent.
    # 
    # #### CHAOS


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


    # ### Marker score
    # 
    # Marker scores evaluate the accuracy of modality fusion and the preservation of biological spatial patterns.
    # 
    # #### Moran's I and Geary's C
    # 
    # Moran's I — measures positive spatial autocorrelation; higher (→1) means marker genes cluster spatially as expected.
    # Geary's C — complementary to Moran's I; lower (→0) indicates stronger spatial clustering.
    # 
    # Runs a Wilcoxon rank test, picks the top n markers per cluster, then computes spatial autocorrelation on those genes.
    # 
    # 


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


    # #### specificity
    # 
    # The specificity score quantifies how specific the expression of a gene, or the intensity of a metabolite is specific to particular spatial clusters within a dataset


    specificity_SM = calculate_gene_specificity(adata_SM, domain_key=PRED_KEY, layers="counts")
    specificity_ST = calculate_gene_specificity(adata_ST, domain_key=PRED_KEY, layers="counts")

    print(f"SM  specificity: {specificity_SM:.4f}  |  ST specificity: {specificity_ST:.4f}")



    # #### logistic feature importance
    # 
    # An effective spatial clustering method should consider feature characteristics, enabling accurate prediction of spatial clusters based on feature expression. The logistic score evaluates this predictive capability.


    logistic_ST = logistic_regression_feature_importance(adata_ST, domain_key=PRED_KEY, layers="counts")
    logistic_SM = logistic_regression_feature_importance(adata_SM, domain_key=PRED_KEY, layers="counts")

    print(f"SM  feature importance: {logistic_SM:.4f}  |  ST feature importance: {logistic_ST:.4f}")



    # #### Mutual information
    # 
    # The Mutual Information Score measures the dependency between feature expression levels and spatial cluster labels


    mi_ST = calculate_mutual_information(adata_ST, domain_key=PRED_KEY, layers="counts")
    mi_SM = calculate_mutual_information(adata_SM, domain_key=PRED_KEY, layers="counts")

    print(f"SM  mutual information: {logistic_SM:.4f}  |  ST mutual information: {logistic_ST:.4f}")



    # ### Biological conservation
    # 
    # To assess the conservation of biological information, metrics such as ARI, NMI, Cell type ASW, Isolated label silhouette, and Graph cLISI score, as defined in scIB45, were utilized.
    # 
    # #### ARI & NMI
    # 
    # Ground truth needed! Annotation of histology images or hand annoted ground truth key needed.
    # 


    ari = compute_ARI(joint_adata, gt_key=PRED_KEY, pred_key=PRED_KEY)
    nmi = compute_NMI(joint_adata, gt_key=PRED_KEY, pred_key=PRED_KEY)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    # Both range 0–1, higher is better


    # #### Avarage Silhoutte Width (ASW)
    # 
    # 
    # 


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


    # #### cLISI


    clisi = compute_clisi_graph(
        joint_adata,
        gt_key=PRED_KEY,
        embedding_key=EMB_KEY
    )
    print(f"cLISI: {clisi:.4f}")
    # Higher = cell-type neighborhoods are purer in embedding space


    # #### Isolated Labels ASW


    isolated_asw = compute_isolated_aws(
        joint_adata,
        gt_key=PRED_KEY,
        embedding_key=EMB_KEY
    )
    print(f"Isolated label ASW: {isolated_asw:.4f}")
    # Higher = rare cell types are well-separated in embedding space


    # ### Results benchmarking


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

    # results = pd.DataFrame(results, index=['spatialMETA'])

    # results_metrics = pd.DataFrame(results_metrics, index=['spatialMETA'])

    return results, results_metrics

#####

import time
import pandas as pd

results_list = []
results_metrics_list = []

for iteration in range(2):
    print(f"starting run no. {iteration}")
    start = time.perf_counter()

    results_run, results_metrics_run = run_vertical_spatialmeta(iteration)

    runtime = time.perf_counter() - start

    results_metrics_run["runtime"] = runtime

    results_list.append(results_run)
    results_metrics_list.append(results_metrics_run)

    print(f"completed run no. {iteration}")

results_df = pd.DataFrame(results_metrics_list)
results_metrics_df = pd.DataFrame(results_metrics_list)

# results_list.to_csv("benchmarks/metrics/automated/results_list.csv", index=False)

results_df.to_excel("benchmarks/results/automated/results.xlsx", index=False)
results_metrics_df.to_excel("benchmarks/results/automated/results_metrics.xlsx", index=False)

print(results_metrics_df)
print("succes")
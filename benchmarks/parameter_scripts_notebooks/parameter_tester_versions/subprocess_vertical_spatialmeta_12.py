import argparse
import json
import numpy as np
import time
from memory_profiler import memory_usage

def run_vertical_spatialmeta(iteration_dir, n_latent, hidden_stacks, lr, max_epoch, n_top_genes, n_top_metabolites, kl_weight, mmd_weight):

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

    # mouse_adata = smt.data.load_adata(
    #     sample_name="Y7_T_raw.aligned",
    #     modality="joint"
    # )

    # mouse_adata.write_h5ad("SpatialMETA/data/Y7_T_mouse.h5ad")

    start_prepro_fit = time.perf_counter()

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
                                            n_top_genes = n_top_genes,
                                            n_top_metabolites = n_top_metabolites,
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
        n_latent = n_latent,
        hidden_stacks = hidden_stacks
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
        max_epoch=max_epoch,
        lr=lr,
        mode='single',
        random_seed = 12,
        kl_weight = kl_weight,
        mmd_weight = mmd_weight
    )

    runtime_prepro_fit = time.perf_counter() - start_prepro_fit


    # ## Loss visualisation


    # In this section, we visualize the loss during the training of the ConditionalVAE model. The loss consists of three components: Kullback-Leibler divergence loss, reconstruction loss for spatial metabolomics (SM), and reconstruction loss for spatial transcriptomics (ST).


    fig,axes = plt.subplots(3, 3, figsize=(20,10))
    axes = axes.flatten()

    for ax, (k, v) in zip(axes, loss_dict.items()):
        ax.plot(v)
        ax.set_title(k)

    plt.tight_layout()
    plt.savefig(f"{iteration_dir}/loss_visualization.png", bbox_inches="tight")
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
        show=False
    )
    plt.savefig(f"{iteration_dir}/umap.png", bbox_inches="tight")

    # ### Hires plots


    # Scatter plot in spatial coordinates.
    # https://scanpy.scverse.org/en/stable/api/generated/scanpy.pl.spatial.html#scanpy.pl.spatial


    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color=["VAE_clusters_latent10"],
        show=False
    )
    plt.savefig(f"{iteration_dir}/spatial.png", bbox_inches="tight")

    # ### Normalized plots 1 (purple)


    sc.pl.spatial(joint_adata,
                img_key="hires",
                color_map = "vlag",
                color=["CD8A","137.04580812911064"],
                layer="normalized",
                show=False)
    plt.savefig(f"{iteration_dir}/normalized1.png", bbox_inches="tight")


    # ### Recondstructed plots


    sc.pl.spatial(joint_adata,
                img_key="hires",
                color_map = "vlag",
                color=["CD8A","137.04580812911064"],
                layer="reconstruction",
                show=False)
    plt.savefig(f"{iteration_dir}/reconstructed.png", bbox_inches="tight")


    # ### Normalized plots 2 (cyan)


    sc.pl.spatial(
        joint_adata,
        img_key="hires",
        color_map = smt.pl.make_colormap(['#2ec4b6','#ffffff','#ff9f1c' ]),
        color=['contribution_st','contribution_sm'],
        layer="normalized",
        alpha_img=0.1,
        size=1.5,
        show=False
    )
    plt.savefig(f"{iteration_dir}/normalized2.png", bbox_inches="tight")

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
    plt.savefig(f"{iteration_dir}/violin_plot.png", bbox_inches="tight")
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



    # Add ground truth label to data


    gt_adata = sc.read_h5ad("data/06_spatialmeta_groundtruth/adata_joint_Y7_T_hvf2800.h5ad")

    # Build mapping on array_row + array_col instead
    gt_annotation = gt_adata.obs[['array_row', 'array_col', 'pathological_annotation']].copy()
    gt_annotation['grid_key'] = (gt_annotation['array_row'].astype(str) + '_' + 
                                gt_annotation['array_col'].astype(str))
    gt_annotation = gt_annotation.set_index('grid_key')

    # Same key on your data
    joint_adata.obs['grid_key'] = (joint_adata.obs['array_row'].astype(str) + '_' + 
                                    joint_adata.obs['array_col'].astype(str))

    # Transfer annotation
    joint_adata.obs['pathological_annotation'] = (
        joint_adata.obs['grid_key']
                .map(gt_annotation['pathological_annotation'])
    )
    joint_adata.obs['pathological_annotation'] = (
        joint_adata.obs['pathological_annotation'].astype('category')
    )

    # Clean up the helper column
    joint_adata.obs.drop(columns=['grid_key'], inplace=True)

    print(joint_adata.obs['pathological_annotation'].isna().sum(), "spots without annotation")
    print(joint_adata.obs['pathological_annotation'].value_counts())


    # Missing spots in ground truth


    missing_mask = joint_adata.obs['pathological_annotation'].isna()

    # # Plot their spatial location
    # sc.pl.spatial(joint_adata, color='pathological_annotation', 
    #             na_color='red', title='Red = unannotated spots')


    sc.pl.spatial(gt_adata, color='pathological_annotation', 
                na_color='red', title='Ground truth')


    missing_mask = joint_adata.obs['pathological_annotation'].isna()
    joint_adata_annotated = joint_adata[~missing_mask].copy()

    # Remove NaN from the category levels themselves (categorical dtype retains them)
    joint_adata_annotated.obs['pathological_annotation'] = (
        joint_adata_annotated.obs['pathological_annotation']
                            .cat.remove_unused_categories()
    )

    print(f"Spots for GT metrics: {joint_adata_annotated.n_obs}")
    print(joint_adata_annotated.obs['pathological_annotation'].value_counts())


    # ### Reconstruction accuracy
    # 
    # PCC & cosine similarity


    import numpy as np
    import scipy.sparse as sp
    from scipy.stats import pearsonr

    def compute_mean_pcc(adata, original_layer='counts', reconstruction_layer='reconstruction'):
        """
        Compute mean Pearson Correlation Coefficient across features (genes/metabolites).
        
        For each feature, correlates its expression vector across all spots
        between the original and reconstructed matrices, then takes the mean.
        
        Parameters
        ----------
        adata : AnnData
        original_layer : str
            Layer containing raw/original counts
        reconstruction_layer : str
            Layer containing VAE reconstructed values

        Returns
        -------
        float : mean PCC across all features
        """
        def to_dense(X):
            return X.toarray() if sp.issparse(X) else np.array(X)

        X_orig  = to_dense(adata.layers[original_layer])   # shape: (n_spots, n_features)
        X_recon = to_dense(adata.layers[reconstruction_layer])

        # Correlate column-wise (per feature across all spots)
        pccs = []
        for j in range(X_orig.shape[1]):
            feature_orig  = X_orig[:, j]
            feature_recon = X_recon[:, j]

            # Skip constant features — pearsonr is undefined for zero-variance vectors
            if feature_orig.std() == 0 or feature_recon.std() == 0:
                continue

            r, _ = pearsonr(feature_orig, feature_recon)
            pccs.append(r)

        n_skipped = X_orig.shape[1] - len(pccs)
        if n_skipped > 0:
            print(f"Skipped {n_skipped} constant features (zero variance)")

        mean_pcc = float(np.nanmean(pccs))
        print(f"Mean PCC across {len(pccs)} features: {mean_pcc:.4f}")
        return mean_pcc


    from sklearn.metrics.pairwise import cosine_similarity

    def compute_mean_cs(adata, original_layer='counts', reconstruction_layer='reconstruction'):
        """
        Compute mean Cosine Similarity across features (genes/metabolites).
        
        For each feature, computes cosine similarity between its expression 
        vector across all spots in the original vs reconstructed matrices,
        then takes the mean.
        
        Parameters
        ----------
        adata : AnnData
        original_layer : str
            Layer containing raw/original counts
        reconstruction_layer : str
            Layer containing VAE reconstructed values

        Returns
        -------
        float : mean cosine similarity across all features
        """
        def to_dense(X):
            return X.toarray() if sp.issparse(X) else np.array(X)

        X_orig  = to_dense(adata.layers[original_layer])   # shape: (n_spots, n_features)
        X_recon = to_dense(adata.layers[reconstruction_layer])

        # Transpose so each row is a feature vector across spots: (n_features, n_spots)
        X_orig_T  = X_orig.T
        X_recon_T = X_recon.T

        similarities = []
        for j in range(X_orig_T.shape[0]):
            feature_orig  = X_orig_T[j].reshape(1, -1)
            feature_recon = X_recon_T[j].reshape(1, -1)

            # Skip zero vectors — cosine similarity undefined
            if np.linalg.norm(feature_orig) == 0 or np.linalg.norm(feature_recon) == 0:
                continue

            cs = cosine_similarity(feature_orig, feature_recon)[0, 0]
            similarities.append(cs)

        n_skipped = X_orig_T.shape[0] - len(similarities)
        if n_skipped > 0:
            print(f"Skipped {n_skipped} zero-vector features")

        mean_cs = float(np.nanmean(similarities))
        print(f"Mean Cosine Similarity across {len(similarities)} features: {mean_cs:.4f}")
        
        return mean_cs


    mean_pcc = compute_mean_pcc(joint_adata)
    mean_cs  = compute_mean_cs(joint_adata)

    print(f"\nReconstruction Accuracy:")
    print(f"  PCC : {mean_pcc:.4f}")
    print(f"  CS  : {mean_cs:.4f}")
    print(f"  Reconstruction accuracy score: {(mean_pcc + mean_cs) / 2:.4f}")


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
        spatial_key='spatial'
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
    # Moran's I measures positive spatial autocorrelation a higher score (→1) means marker genes cluster spatially as expected.
    # Geary's C just as Moran's I, a lower value (→0) indicates stronger spatial clustering.
    # 
    # Runs a Wilcoxon rank test, picks the top n markers per cluster, then computes spatial autocorrelation on those genes or metabolites.


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

    moranI, gearyC = marker_score(joint_adata, domain_key=PRED_KEY, top_n=50)
    print(f"Combined Moran's I: {moranI:.4f} | Combined, Geary's C: {gearyC}")


    # #### specificity
    # 
    # The specificity score quantifies how specific the expression of a gene, or the intensity of a metabolite is to each  spatial clusters within a dataset. The returned score is the most specific feature-cluster combination.


    specificity_SM = calculate_gene_specificity(adata_SM, domain_key=PRED_KEY, layers="counts")
    specificity_ST = calculate_gene_specificity(adata_ST, domain_key=PRED_KEY, layers="counts")

    print(f"SM  specificity: {specificity_SM:.4f}  |  ST specificity: {specificity_ST:.4f}")

    specificity = calculate_gene_specificity(joint_adata, domain_key=PRED_KEY, layers="counts")

    print(f"Combined specificity: {specificity:.4f}")


    # #### logistic feature importance
    # 
    # The logistic score evaluates the predictive capability of feature characteristics to predict clusters based on it's feature expression.


    logistic_ST = logistic_regression_feature_importance(adata_ST, domain_key=PRED_KEY, layers="counts")
    logistic_SM = logistic_regression_feature_importance(adata_SM, domain_key=PRED_KEY, layers="counts")

    print(f"SM logistic feature importance: {logistic_SM:.4f}  |  ST feature importance: {logistic_ST:.4f}")

    logistic = logistic_regression_feature_importance(joint_adata, domain_key=PRED_KEY, layers="counts")

    print(f"Combined logistic feature importance: {logistic:.4f}")



    # #### Mutual information
    # 
    # The Mutual Information Score measures the dependency between feature expression levels and spatial cluster labels


    mi_ST = calculate_mutual_information(adata_ST, domain_key=PRED_KEY, layers="counts")
    mi_SM = calculate_mutual_information(adata_SM, domain_key=PRED_KEY, layers="counts")

    mi = calculate_mutual_information(joint_adata, domain_key=PRED_KEY, layers="counts")
    print(f"SM  mutual information: {logistic_SM:.4f}  |  ST mutual information: {logistic_ST:.4f} | Combined mutual information: {mi}")



    # ### Biological conservation
    # 
    # To assess the conservation of biological information, metrics such as ARI, NMI, Cell type ASW and Graph cLISI score, as defined in scIB45, were utilized.
    # 
    # #### ARI & NMI
    # 
    # Ground truth needed! Ground truth has been manually annoted using H&E staining.


    ari = compute_ARI(joint_adata_annotated, gt_key=GT_KEY, pred_key=PRED_KEY)
    nmi = compute_NMI(joint_adata_annotated, gt_key=GT_KEY, pred_key=PRED_KEY)
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    # Both range 0–1, higher is better


    # #### Avarage Silhoutte Width (ASW)
    # 
    #  Instead it tries to estimate the quality of the clustering by how much each object is like its own cluster compared to other clusters. A high ASW indicates that each cluster is well defined and very different from other clusters, while a low ASW indicates that objects in clusters are not very different from each other. 


    asw = compute_ASW(
        joint_adata,
        pred_key=PRED_KEY,
        spatial_key='spatial'
    )
    print(f"Spatial ASW: {asw:.4f}")


    # #### cLISI
    # 
    # This quality metric quantifies the cell type purity in a local neighborhood. It achieves this by determining if the neighbours of each spot are the same as the spot itself.


    clisi = compute_clisi_graph(
        joint_adata_annotated,
        gt_key=GT_KEY,
        embedding_key=EMB_KEY
    )
    print(f"cLISI: {clisi:.4f}")
    # Higher = cell-type neighborhoods are purer in embedding space


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
        'ASW':           asw,
        'cLISI':         clisi
    }

    reconstrucion_accuracy = (mean_pcc + mean_cs)/2
    continuity_score = ((1 - chaos_score) + (1 - pas_score))/2
    marker_score_ST = (moranI_ST + (1 - gearyC_ST) + specificity_ST + mi_ST)/4
    marker_score_SM = (moranI_SM + (1 - gearyC_SM) + specificity_SM + mi_SM)/4
    marker_score_combined = (moranI + (1 - gearyC) + logistic + specificity + mi)/5
    biological_conservation = (ari + nmi + asw + clisi)/4

    results_metrics = {"Continuity Score": continuity_score,
                    "Marker score": marker_score_combined,
                    "Marker Score ST": marker_score_ST,
                    "Marker score SM": marker_score_SM,
                    "Biological Conservation": biological_conservation,
                    "Reconstruction Accuracy": reconstrucion_accuracy
                    }

    # code to see what types each of the benchmarking variables are
    # for key, value in results.items():
    #     print(f"{key} = {type(value)}")
    # print()
    # for key, value in results_metrics.items():
    #     print(f"{key} = {type(value)}")

    # results = pd.DataFrame(results, index=['spatialMETA'])

    # results_metrics = pd.DataFrame(results_metrics, index=['spatialMETA'])

    # adding runtime to the dictionary
    results["runtime_preprocessing_fitting"] = runtime_prepro_fit
    results_metrics["runtime_preprocessing_fitting"] = runtime_prepro_fit

    return results, results_metrics

#####

# argparse
parser = argparse.ArgumentParser()

# argparse argumenten lijst
parser.add_argument("-i", "--iteration_directory", type=str, help="Input for where results should be saved for this iteration")
parser.add_argument("-l", "--latent_dimensions", type=int,  help="Input for amount of latent dimensions")
parser.add_argument("-s", "--hidden_stacks", type=int, nargs="+", help="Input for amount of hidden stacks")
parser.add_argument("-r", "--learning_rate", type=float, help="Input for learning rate")
parser.add_argument("-m", "--max_epoch", type=int, help="Input for max_epoch")
parser.add_argument("-g", "--n_top_genes", type=int, help="Input for n_top_genes")
parser.add_argument("-b", "--n_top_metabolites", type=int, help="Input for n_top_metabolites")
parser.add_argument("-k", "--kl_weight", type=float, help="Input for kl_weight")
parser.add_argument("-d", "--mmd_weight", type=float, help="Input for mmd_weight")

# argumenten opslaan als variabelen
args = parser.parse_args()

iteration_dir = args.iteration_directory
n_latent = args.latent_dimensions
hidden_stacks = args.hidden_stacks
lr = args.learning_rate
max_epoch = args.max_epoch
n_top_genes = args.n_top_genes
n_top_metabolites = args.n_top_metabolites
kl_weight = args.kl_weight
mmd_weight = args.mmd_weight

# timer
start = time.perf_counter()

# Run SpatialMETA (without tracking maximum memory usage)
# results_run, results_metrics_run = run_vertical_spatialmeta(iteration_dir, n_latent, hidden_stacks)

# Run SpatialMETA
peak_memory, results = memory_usage((run_vertical_spatialmeta, (), {"iteration_dir": iteration_dir, "n_latent": n_latent, "hidden_stacks": hidden_stacks, "lr": lr, "max_epoch": max_epoch, "n_top_genes": n_top_genes, "n_top_metabolites": n_top_metabolites, "kl_weight": kl_weight, "mmd_weight": mmd_weight}), retval=True, max_usage=True)

runtime = time.perf_counter() - start

results_run, results_metrics_run = results

# omzetten naar normale floats ipv numpy floats
results_run = {k: float(v) if isinstance(v, np.floating) else v
           for k, v in results_run.items()}

results_metrics_run = {k: float(v) if isinstance(v, np.floating) else v
           for k, v in results_metrics_run.items()}

# adding runtime to the dictionary
results_run["runtime"] = runtime
results_metrics_run["runtime"] = runtime

# adding peak_memory to the dictionary
results_run["peak_memory"] = peak_memory
results_metrics_run["peak_memory"] = peak_memory

# wegschrijven van resultaten om ze in te laden in main script
with open("results_run.json", "w") as f:
    json.dump(results_run, f)

with open("results_metrics_run.json", "w") as f:
    json.dump(results_metrics_run, f)

print("Hier onder type van results metrics run")
print(type(results_metrics_run))
print("subproces python proces succes")
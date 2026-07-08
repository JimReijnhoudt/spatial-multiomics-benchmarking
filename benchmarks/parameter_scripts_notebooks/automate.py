print("succes")
for i in range(2):
    print(i)

value = 1e-4
print(str(value))

# import anndata as ad

# adata = ad.read_h5ad(".conda/envs/spatial/lib/python3.9/site-packages/spatialmeta/data/datasets/adata_SM_Y7_T_raw_raw.h5ad")

# print(adata.shape)

# adata = ad.read_h5ad(".conda/envs/spatial/lib/python3.9/site-packages/spatialmeta/data/datasets/adata_ST_Y7_T_raw_raw.h5ad")

# print(adata.shape)

list = [128, 256]
print(str(list))

# #
# parameter_values = {
#     "n_latent": 10,
#     "hidden_stacks": [128],
#     "lr": 1e-3,
#     "max_epoch": 64,
#     "n_top_genes": 2000,
#     "n_top_metabolites": 800,
#     "kl_weight": 1,
#     "mmd_weight": 1,
# }
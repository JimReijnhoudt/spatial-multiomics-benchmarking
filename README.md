# Spatial Multi-Omics Docker Setup

This Docker image contains two conda environments for spatial multi-omics analysis:

| Environment | Python | Packages |
|---|---|---|
| `spatial` | 3.9 | SpatialMETA, Scanpy, Squidpy |
| `miso` | 3.7 | MISO, Scanpy, PyTorch |

---

## Requirements

- Docker
- NVIDIA Container Toolkit (for GPU support)
- VS Code with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)

---

## Build

From the directory containing the Dockerfile:

```bash
docker build -t spatial-multiomics .
```

Build takes approximately 20-30 minutes due to package downloads.

---

## Run

**With GPU:**
```bash
docker run --gpus all -dit -v $(pwd):/workspace --name spatial-container spatial-multiomics
```

**Without GPU:**
```bash
docker run -dit -v $(pwd):/workspace --name spatial-container spatial-multiomics
```

The `-d` flag runs the container in the background so it stays alive independently
of your terminal session. Stop it later with:

```bash
docker stop spatial-container
```

---

## Connecting in VS Code

1. Open VS Code
2. Press `Ctrl+Shift+P` and run **Dev Containers: Attach to Running Container**
3. Select `spatial-container` from the list
4. A new VS Code window opens inside the container
5. Open `/workspace` as your folder: **File → Open Folder → /workspace**

---

## Selecting a Kernel in Notebooks

When opening a `.ipynb` file, click the kernel picker in the top-right corner
of VS Code and select either:

| Kernel | Use for |
|---|---|
| `spatial` | SpatialMETA notebooks |
| `miso` | MISO notebooks |

---

## Saving Data

The `/workspace` directory is mounted to your local machine. Always save data
here so it persists after the container stops:

```python
import os
os.makedirs('/workspace/data', exist_ok=True)

# Save AnnData objects
adata.write_h5ad('/workspace/data/my_data.h5ad')
```

Files saved anywhere else inside the container (e.g. `/opt/`, `/tmp/`) will be
lost when the container is removed.

---

## Downloading SpatialMETA Datasets

Datasets from the SpatialMETA paper are downloaded from Zenodo on first use:

```python
import spatialmeta as smt

# See available datasets
print(smt.data.list_datasets())

# Download the primary example dataset (~111MB)
adata = smt.data.load_adata('adata_joint_Y7_T_raw.h5ad')

# Save to workspace so it doesn't need to be re-downloaded
os.makedirs('/workspace/data', exist_ok=True)
adata.write_h5ad('/workspace/data/Y7_T_joint.h5ad')
```

Once saved to `/workspace/data/`, load it directly next time:

```python
import scanpy as sc
adata = sc.read_h5ad('/workspace/data/Y7_T_joint.h5ad')
```

---

## Downloading H&E image for MISO 3m

MISO 3M requires the raw H&E image to generate an image embedding. Because the raw image file is too large to include in this repository, you can download it here:
[Download the raw H&E image](https://figshare.com/articles/dataset/Multi-omic_profiling_of_clear_cell_renal_cell_carcinoma_identifies_metabolic_reprogramming_associated_with_disease_progression/24599295?file=43225668)

---

## CRITICAL: MISO Pretrained Weights

Sometimes the MISO GitHub repository exceeds its LFS storage budget, meaning the
pretrained ViT weights cannot be downloaded automatically during the build.

If that is the case you must copy them manually into the running container:

**Step 1 — Copy into the running container:**
```bash
docker cp vit4k_xs_dino.pth spatial-container:/opt/miso/miso/checkpoints/
docker cp vit256_small_dino.pth spatial-container:/opt/miso/miso/checkpoints/
```

**Step 2 — Commit the container so weights persist across restarts:**
```bash
docker commit spatial-container spatial-multiomics:with-weights
```

From now on use `spatial-multiomics:with-weights` as your image name instead
of `spatial-multiomics`.

---

## Confirmed Working Package Versions

### spatial environment
| Package | Version |
|---|---|
| Python | 3.9 |
| numpy | 1.24.4 |
| pandas | 1.5.3 |
| matplotlib | 3.5.3 |
| matplotlib-inline | 0.1.6 |
| scanpy | 1.9.1 |
| squidpy | 1.2.2 |
| dask | 2021.10.0 |
| xarray | 2024.2.0 |
| torch | 2.2.2+cu121 |
| spatialMETA | 0.0.3.0 |

### miso environment
| Package | Version |
|---|---|
| Python | 3.7 |
| numpy | 1.21.6 |
| pandas | 1.3.5 |
| matplotlib | 3.5.3 |
| matplotlib-inline | 0.1.6 |
| scanpy | 1.9.1 |
| torch | 1.13.1 |
| torchvision | 0.14.1 |
| scipy | 1.7.3 |
| scikit-learn | 1.0.2 |
| opencv-python | 4.7.0.72 |

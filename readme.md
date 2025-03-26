# Identifying Patterns of Tumour Tissue Architecture with Deep Representation Learning

**Part of Level 4 Individual Project at the University of Glasgow.** \
**by Jetanat Sihanatkathakul 2923241s**

This repository contains code files, scripts, and a portion of data related to the
dissertation titled _"Identifying Patterns of Tumour Tissue Architecture 
with Deep Representation Learning"_.

[GitHub Repository](https://github.com/crystilonz/tumour_deep_learning)

---
## Research Goals
This project covers two different but interconnected questions:
1. How effectively can a model that utilises deep representations of whole slide images presented in the Histomorphological Phenotype Learning (HPL) 
methodology classify cancerous WSIs to the correct cancer type?

2. How can tile-level annotations be generated, and can these annotations be used to train a
model that produces descriptive captions for lung cancer tiles?

---

## Repository Structure

```
tumour_deep_learning/                       
├─ data/                         
├─ dissertation/
├─ presentation/
├─ research/
├─ src/
│  ├─ data_manipulation/
│  ├─ datasets/
│  ├─ models/
│  │  ├─ application/
│  │  ├─ interface/
│  ├─ script_info/
│  ├─ scripts/
│  ├─ utils/
```

* `./data` contains the raw data or scripts/notebooks related to data acquisition.
Most of the codes are related to the [DeepPATH](https://github.com/ncoudray/DeepPATH) framework 
used to tile whole slide images, or the [Histomorphological Phenotype Learning (HPL)](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master)
methodology.
* `./dissertation` contains the original LaTeX dissertation file along with
the figures and other accompanying files.
* `./presentation` contains files related to the video presentation.
* `./research` contains some research notes from reading other publications.
* `./src` is the most critical directory. It contains the majority of the codes and scripts used to
produce the results and write the dissertation.

Under the `src/` directory:
* `./src/data_manipulation/` includes files related to data processing and transformation.
* `./src/datasets/` includes the processed dataset, or intermediate data under processing. The files serve as an input for
further manipulation with data pipelines or directly to the models.
* `./src/models/` contains model implementations in PyTorch. Codes that build on these models can be found
in the `application/` subdirectory. The `interface/` subdirectory contains interface definitions, primarily for cleaner code structure.
* `./src/script_info/` is used to store information (e.g., file directories) common between scripts.
* `./src/scripts/` stores the scripts to run parts of the project. Scripts are written in slurm code, or Python code.
* `./src/utils/` contains various supporting functions, such as model training and plotting functions.

---

## Results

> Additional files are generally available on [Google Drive](https://drive.google.com/drive/folders/1Mr_wf4Xu57bDxAumkZ4_uaK4q_x9QKRQ?usp=sharing).

The model checkpoints and raw results used for analyses in the dissertation
are available in this [link](https://drive.google.com/drive/folders/18rlcH_Hv0qdwBbuw2ss50W5F4CKC6I-z?usp=share_link).

The contents of the directories in the link above are as follows:
* `saved_models/` contains checkpoints and results used for development. The results in this directory
are *not* included in the dissertation.
* `validation_models/` contains the models and the results of 5-fold cross-validation on the TCGA dataset.
The results of the cancer classifiers regarding the first research question use these checkpoints.
* `dist_models/` includes the checkpoints for CNN-RNN tile captioning models after the distributed training process
on the dirac tursa cluster. The `dev` directories are development models, which are not included in the dissertation.

Other directories do not include any checkpoints, but are results from evaluating the frozen model on different datasets:
* `gtex_validation_results/`: classifier models assessed on the external (CMB + CPTAC) dataset.
* `external_validation_results/`: classifier models assessed on the GTEx dataset (OoD testing).

---

## Reproduction

With the repository as is, partial reproduction is possible, such as training a new model on the dataset.
Further reproduction will require additional data files, some of which are quite bulky. 

Please refer to the [user manual](src/manual.md) for details.


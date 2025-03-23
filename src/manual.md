# User Manual 

This documentation describes how to run the code related to the paper for reproduction purposes.

> Most of the code was tested and run on Apple Silicon MacOS, using the non-cuda version of PyTorch.

---

## Getting Started

This project requires `Python 3.12` to run. Major dependencies include:

* PyTorch
* NumPy
* pandas
* scikit-learn
* Torchmetrics
* matplotlib
* seaborn
* SHAP
* spaCy
* h5py

Install dependencies by running this command:
```
pip install -r requirements.txt 
```

These dependencies only cover code files in this project. Data acquisition scripts work with
external repositories, like [DeepPATH](https://github.com/ncoudray/DeepPATH) and 
[HPL](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master). 
Please refer to the documentations in these repositories to setup the workspace.

The scripts required (and some more) are contained in `scripts/` directory. There are also some associated 
`bash` files that help run these scripts.

---

## Bash Files

`.sh` files assume Python environment paths managed by PyCharm IDE (i.e. `./LinuxBASH/bin/python`).
If this is not the case for you, change the Python path to your preferred environment.
 
For example, you can change line 3`src/train_all.sh` to use the system's Python interpreter:

```
for f in scripts/training_scripts/*.py; do python3 "$f"; done
```

These `bash` scripts simply run all Python files in a specific directory inside `src/scripts/`, depending on the task (some do a litte bit more). 
You can always run each python script individually. These `bash` scripts facilitate 
running files in bulk.

> Make sure you are in the `src/` directory before running the bash files.

---

## File Path

> **Important:** Python scripts/functions will often expect *absolute paths* as arguments. 
> 
Before running files and notebooks, paths are required to be modified to point to the correct files. In particular, evaluation
scripts will require paths to the model checkpoint file, and data manipulation scripts will require paths to appropriate data files.

Training scripts generally do not require paths. It will produce results and checkpoints under some directories
inside `src/`.

---

## Data Files and Checkpoints

The data provided in the repository will allow you to train and evaluate the cancer classifiers.
It will not allow you to train or evaluate the text captioning model. It is also not enough to run notebooks within the `src/models/application`
directory, as they require the raw tile pictures.

### Tile-Caption Data
The complete dataset is available here: https://drive.google.com/drive/folders/15PEVjwzNdl41XuXgMsdkJAJhSiHOBGN4?usp=sharing

Download the file and replace the `src/datasets` directory with the downloaded one.
This will allow you to train and evaluate LSTMs.

### Checkpoints
The checkpoints (and the results) are available here: https://drive.google.com/drive/folders/18rlcH_Hv0qdwBbuw2ss50W5F4CKC6I-z?usp=share_link

It is recommended that you put the directories in the Google Drive inside `src/` to immitate the behaviour of the
scripts. It is also recommended if you want to reproduce the text captioning part of the project,
as the CNN-RNN takes very long to train, and the training script (for the model as in the dissertation) is only available
as a distributed training script, intended to be run on the dirac tursa cluster.

### Raw Data
> These files are provided by the HPL repository.

This file contains the [HDF5 raw images](https://drive.google.com/file/d/1rxuum9_rk1UoE3rphWJ-skAzYpkk1F_J/view?usp=share_link).
It is recommended to be saved to `data/TCGA_Lung/hdf5_TCGAFFPE_LUADLUSC_5x_60pc_he_validation.h5`.

This file contains the [raw vectors for tile images from HPL:](https://drive.google.com/file/d/1KEHA0-AhxQsP_lQE06Jc5S8rzBkfKllV/view).
It should be saved to `data/TCGA_Lung/TCGA_Lung.h5`.

---

## Training Classifiers
### Normally
These models are not included in the dissertation. You can train the classifer model on a typical train:test split by either:
* Running `src/train_all.sh`
* Running files in `src/scripts/training_scripts/`

The models and the validation results will be saved under `src/saved_models/` directory.

### 5-fold cross-validation
This corresponds to the models and the results in the dissertation. You can do this by:
* Running `src/validate_all.sh`
* Running files in `src/scripts/validating_scripts/`

The best models across from 5 folds and the results will be saved under `src/validation_models/`.
The `bash` file will run metrics aggregation for you. If you are running the files manually, then
you can run `src/utils/plot_validation.py` file to generate the aggregated plot.

---

## Evaluating Classifiers
> Remember that these scripts require you to change the paths to the checkpoints as _absolute paths_.

### External Dataset
You are required to change the paths in the scripts inside `src/scripts/external_evaluation_scripts` to your model checkpoint locations.
You can perform classifier evaluation on the external dataset by either:
* Running `src/external_evaluation_all.sh`
* Running files in `src/scripts/external_evaluation_scripts`

The results will be saved in `src/external_validation_results/`.

### GTEx Dataset
You are required to change the paths in the scripts inside `src/scripts/GTEX_evaluation_scripts` to your model checkpoint locations.
You can perform classifier evaluation on the GTEx dataset by either:

* Running `src/gtex_evaluation_all.sh`
* Running files in `src/scripts/GTEX_evaluation_scripts`

---

## Training the Tile Captioning Models
The tile captioning models as presented in the dissertation were trained on the dirac tursa cluster, using 
distributed training methods. You can try running this command:

```
python pytorch_run_lstm.py --help
```

to see the arguments. The slurm scripts used to train the model presented in the dissertation is located inside 
`src/scripts/slurm_scripts`.

You can train the CNN-RNN model locally by running `src/train_lstm.py`. This will train the z-vector LSTM model
for 100 epochs. The model and the results from testing will be saved at `src/saved_models/LSTM/`.

---

## Evaluating the Tile Captioning Models
If you downloaded the checkpoints from [here](https://drive.google.com/drive/folders/18rlcH_Hv0qdwBbuw2ss50W5F4CKC6I-z)
you can evaluate it with the original test set (inside the pickle files downloaded from [here](https://drive.google.com/drive/folders/15PEVjwzNdl41XuXgMsdkJAJhSiHOBGN4))
by running the scripts inside `src/scripts/LSTM_evaluation_scripts`.

> Make sure you provide the appropriate checkpoints to the scripts: h-vector checkpoint for the h-vector script, etc.

Results will be saved at the same location as the model checkpoint's.

---

## Captioning Images
`src/models/application/caption_lstm.ipynb` notebook provides functions to caption a random image in the HDF5 file. Make sure
you download the HDF5 file containing the images from [this link](https://drive.google.com/file/d/1rxuum9_rk1UoE3rphWJ-skAzYpkk1F_J/view?usp=share_link).
You will also need to change the paths near the top of the notebook for it to work.





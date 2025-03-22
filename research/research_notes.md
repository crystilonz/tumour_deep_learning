# Research Notes
This file contains notes from literature reviews, with its references.

### Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides
> https://www.nature.com/articles/s41467-024-48666-7

    @article{Claudio2024Mapping,
	journal = {Nature Communications},
	doi = {10.1038/s41467-024-48666-7},
	issn = {2041-1723},
	number = {1},
	language = {en},
	publisher = {Springer Science and Business Media LLC},
	title = {Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides},
	url = {http://dx.doi.org/10.1038/s41467-024-48666-7},
	volume = {15},
	author = {Claudio Quiros, Adalberto and Coudray, Nicolas and Yeaton, Anna and Yang, Xinyu and Liu, Bojing and Le, Hortense and Chiriboga, Luis and Karimkhan, Afreen and Narula, Navneet and Moore, David A. and Park, Christopher Y. and Pass, Harvey and Moreira, Andre L. and Le Quesne, John and Tsirigos, Aristotelis and Yuan, Ke},
	date = {2024-06-11},
	year = {2024},
	month = {6},
	day = {11},}

- Proposing new self-learning based method for WSI
- From whole slide, divide into tiles of 224x224 using `DeepPATH`
  - Tissue with more than 60% whitespace is discarded
  - Stain normalization with `Reinhard's Method`
- From each slide, use self-supervised learning to try to change one slide to a vector representation fo size 128
- Use Barlow-Twin method to train the model
  - Augment data with a pipeline
  - Train backbone network --> a mix of:
    - CNN: several ResNet layers --> 19 CNN layers
    - Self-attention layer --> 1 layer
    - more details in `Quiros et al.`
  - 250K tile images --> 60 epochs, batch size 64
- After training, the model is frozen and then used to turn slide images into vector representations
- The model is trained on lung samples: LUSC and LUAD samples from TCA
- The dataset is turned into a graph, and then clustering is performed on the graph
  - Each tile is transformed to a tile vector representation (size $R^D, D=128 $).
  - 200,000 randomly sampled tile vectors from training set is turned into a graph by using k-nearest neighbour ($k=250$)
  - Use the graph above to find the clusters using `Leiden Community`
    - Leiden Resolution or $\gamma$ 

### AI-based pathology predicts origins for cancers of unknown primary

> https://www.nature.com/articles/s41586-021-03512-4

    @article{lu2021ai,
    title={AI-based pathology predicts origins for cancers of unknown primary},
    author={Lu, Ming Y and Chen, Tiffany Y and Williamson, Drew FK and Zhao, Melissa and Shady, Maha and Lipkova, Jana and Mahmood, Faisal},
    journal={Nature},
    volume={594},
    number={7861},
    pages={106--110},
    year={2021},
    publisher={Nature Publishing Group}
    }

- Model utilising Multi-Instance-Learning (MIL)
  - Chop up WSI into tiles (like before)
  - Use pre-trained model based on ResNet50 (truncated ResNet50) which was trained on ImageNet as a feature extractor
  - from ^ we get size 1024 feature vector for each slide
  - from ^ put into 2 fully connected layers (new training) to learn histology specific features
  - We now get a feature vector for each tile in the WSI
  - in each bag (WSI), pool the feature vectors by using attention based pooling
  - after attention based pooling, the entire WSI will be reduced to a vector representation
  - concatenate sex to the vector from ^, then feed to classifier
- Model uses multitask learning
  - First task is where the tumour originates (multiclass classification)
  - Second task is whether the tumour is a primary cancer or metastatic cancer (binary classification)
  - This is performed by having (partly) different sets of weights for two tasks in the attention pooling
  - loss function is defined as a linear combination of losses from both task
    - 0.75(loss from origin classification) + 0.25(loss from metastasis classification)
- Data is a combination of publicly available data and in-house data
  - public data like TCGA
- Classifier is logistic regression


  

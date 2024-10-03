# Research Notes
This file contains notes from literature reviews, with its references.

## Week 1

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
  - m
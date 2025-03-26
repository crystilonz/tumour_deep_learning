# Presentation Script

## Intro

Hello. This is a presentation on my individual project: Identifying Patterns of Tumour
Architecture with Deep Representation Learning. I will present to you the rough overview of the project.
Further details and specifics are available in the dissertation.

The project involves two research questions:

1. How effectively can a model that utilises deep representations of whole slide images presented in the Histomorphological Phenotype Learning (HPL) methodology classify cancerous WSIs to the correct cancer type?

This question deals with classification of the cancer histopathological slide. We want to classify these slides to the correct type of cancer.

2. How can tile-level annotations be generated, and can these annotations be used to train a model that produces descriptive captions for lung cancer tiles?

This question covers building a model that can describe a tile or a view of lung cancer histopathological slide. The
input to the model is basically an image of lung cancer tissue, and the expected output is a short text, describing what is in the image.

Before we move on to the more in-depth topics on the project, I think the introduction of HPL is appropriate.

HPL, or Histomorphological Phenotype Learning, is a methodology presented by Claudio et al. The methodology uses self-supervised
learning algorithms to build a methodology that can cluster tile images into different communities, using Leiden community detection.

This projects builds on this methodology, by using various parts:
1. First, I will use the representation of a whole slide image, constructed from presentation ratios of clusters in the
slide, to perform classification in the first question of the project. The representation is called a Whole Slide Image Vector
2. Second, I will build upon this methodology to create a workflow that can produce tile-level annotation data. This
data will be used to train the tile-captioning model in question 2.


## Whole Slide Image Classification

Next, I will talk about the first question: classifying whole slide images.

As mentioned, for classification purposes, the models will treat the whole slide image vector as the representation of the slide.
The data was from TCGA or The Cancer Genome Atlas project, comprising a total of 10 cancer types.

Multiple classifier architectures were implemented for this experiment. There are two different baseline models:
1. The Logistic Regression model
2. The deep Multilayer Perceptron model which refers to this deep typical modern feedforward neural network

There were other variants of the multilayer perceptron model. The baseline model simply interweaves linear layers with ReLU
activation function layers. The other models aim to improve performance by applying common techniques in machine learning. These involve:
1. Swapping ReLU with leakyReLU
2. Introducing dropout regularisation

5-fold cross-validation was performed on all models with the TCGA dataset. The results indicated that there was no significant difference between models in term of performance.
The performance was not spectacular as you can see on the figures on the right.

Other than the TCGA dataset, I collected slides from other sources and evaluated the model on this external dataset.
The models' performances on this dataset were noticeably lower than their performances on the TCGA dataset.

Additionally, slides were collected from the GTEx source, short for Genetype-Tissue Expression project. Unlike other sources,
slides procured here did not contain cancer tissue. These were slides of normal tissue. I actually assessed all classifier models,
but the results were roughly the same between models, so I will present the leaky version of the multilayer perceptron model here.

The model, which was trained on cancerous tissue slides, was assessed on normal tissue slides as out-of-distribution testing. The model performed decently well on this data. In fact, the performance metrics were roughly comparable to those 
on the external dataset. 

Surprisingly, the models showed systematic confusion patterns. The model seemed to be misclassifying skin samples as breast, and
cervix samples as prostate.

The results indicated that there are limitations to using whole slide image vectors as representations for classification purposes.

The models exhibited limited generalisability, probably because:
1. the models were trained on data from only one source. So the performance dropped when the models tried to classify
slides from different sources. They maybe some fundamental differences between sources.
2. the data contained only primary tumour slides. So the model performs equally well on external dataset and during out-of-distribution testing.

I investigated further from the systematic confusion pattern, by looking at the clusters responsible. It seemed like clustering
from Histomorphological Phenotype Learning can cluster visually similar tile images together, but sometimes these correspond to different tissue structures.
It seems like this was the main culprit for the misclassification patterns.

## Lung Cancer Tile Image Captioning
Next, I will talk about the second research question: captioning a lung cancer tissue tile image

One challenge in the field of histopathological image analysis is the lack of data. In particular, tile-level annotations.
It may be easy to find slide-level annotations, for example "this is a lung cancer slide", but the specifics on "what this particular
this contains" is very scarce.

To tackle this problem, remember that HPL provides means to cluster visually similar tile together. We can produce
annotations for these clusters, as a way to annotate all similar looking tiles at once. This is the main idea
of this data workflow.

The tiles are clustered with HPL, and annotations are made for these clusters. The cluster annotations are then propagated back
to the tiles that are members of that cluster, producing tile-annotation pairings which can be used to train the image captioning model, answering 
the second research question.

The whole picture looks like this. The architecture chosen for the tile captioning model is the CNN-RNN architecture.
1. The Convolutional Neural Network (CNN) is used to encode the tiles to vectors containing relevant feature values.
In this project, I used the CNN from HPL as the encoder, as I had limited computing power and memory.
2. The Recurrent Neural Network (RNN) is used to decode the tile image embedding from the CNN to a natural language text.

Two models were built, using different feature vectors and different hidden sizes. The larger model performed better, so 
the metric values here correspond to those of the larger model. The model was trained and tested on the lung adenocarcinoma dataset from TCGA. I used 
automated metrics to evaluate the model: Bleu and Rouge. Bleu is more similar to precision, while Rouge is to recall. As you
can see, these metrics suggest the model can produce captions that closely match the ground truth annotations.

One important point to mention is these metrics compare the model's output with the tile-annotation pairings as created by the workflow.
Unfortunately, I did not have access to resources to assess the quality of tile-annotation pairings themselves.

Here you can see some examples of the predictions, alongside the annotations. The model captions the tile images that you see,
with a short text description to the right of each image.

This text-captioning model serves as an example, or a proof of concept, that you can produce tile-level annotations, and a model
trained on those annotations using direct supervised-learning methods.

However, further scaling of this workflow requires further quality assurance processes, as this workflow relies heavily on the accuracy
of the annotations.

That is the end of the presentation. Thank you.
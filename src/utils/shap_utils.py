import shap
from pathlib import Path

import sklearn.utils

from models.pan_cancer_classifier import PanCancerClassifier
from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_LABELS
import torch
from matplotlib import pyplot as plt
from random import randrange
from typing import Literal


def shap_waterfall_pancancer(m: torch.nn.Module,
                             data: torch.Tensor,
                             labels: torch.Tensor,
                             e: shap.DeepExplainer = None,
                             correct: Literal["true", "false", "both"] = "true",
                             sample_names=None,
                             slide_names=None):
    if e is None:
        e = shap.DeepExplainer(m, data)

    shap_values = e(data)
    base_values = e.expected_value

    probs = m(data)
    predictions = torch.argmax(probs, dim=1)

    # slicing
    if correct == "true":
        # we only want to get the correct predictions here
        plotting_shap_values = shap_values[labels == predictions]
        plotting_predictions = predictions[labels == predictions]
        plotting_labels = labels[labels == predictions]
        plotting_samples = sample_names[(labels == predictions).tolist()] if sample_names is not None else None
        plotting_slides = slide_names[(labels == predictions).tolist()] if slide_names is not None else None
    elif correct == "false":
        # only wrong samples!
        plotting_shap_values = shap_values[labels != predictions]
        plotting_predictions = predictions[labels != predictions]
        plotting_labels = labels[labels != predictions]
        plotting_samples = sample_names[(labels != predictions).tolist()] if sample_names is not None else None
        plotting_slides = slide_names[(labels != predictions).tolist()] if slide_names is not None else None
    else:
        # getting both
        plotting_shap_values = shap_values
        plotting_predictions = predictions
        plotting_labels = labels
        plotting_samples = sample_names if sample_names is not None else None
        plotting_slides = slide_names if slide_names is not None else None

    fig = plt.figure(figsize=(150, 75))

    # add the horizontal title
    for prediction_class in range(0, 10):
        ax = plt.subplot(10, 1, prediction_class + 1)
        ax.spines['left'].set_position(('axes', -0.02))
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(PAN_CANCER_LABELS[prediction_class], fontsize=48)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    # plot 1 waterfall for each class, total of 10 samples one from each class
    for sample_class in range(0, 10):
        # slice
        shap_values_in_class = plotting_shap_values[plotting_labels == sample_class]
        predictions_in_class = plotting_predictions[plotting_labels == sample_class]

        # random sample
        sample_num = randrange(len(shap_values_in_class))

        # prediction for this sample
        sample_prediction = predictions_in_class[sample_num]

        # get the sample name
        if plotting_slides is not None and plotting_samples is not None:
            samples_in_class = plotting_samples[plotting_labels == sample_class]
            slides_in_class = plotting_slides[plotting_labels == sample_class]
            sample_name = samples_in_class[sample_num].decode('utf-8')
            slide_name = slides_in_class[sample_num].decode('utf-8')
        else:
            sample_name = f"Sample of class {PAN_CANCER_LABELS[sample_class]}"
            slide_name = ""

        # add plot
        col_name = sample_name + '\n' + slide_name
        plt.subplot(1, 10, sample_class + 1)
        plt.title(col_name, fontsize=48, y=1.01)
        plt.axis('off')

        for shap_class in range(0, 10):
            ax = fig.add_subplot(10, 10, shap_class * 10 + sample_class + 1)
            plt.sca(ax)
            # if prediction is correct:
            if sample_prediction == sample_class:
                # label the correct plot with green
                if shap_class == sample_class:
                    ax.set_facecolor('#dbfcba')
            else:  # prediction is wrong
                if shap_class == sample_prediction:
                    # colour prediction plot with red
                    ax.set_facecolor('#ffddd4')
                elif shap_class == sample_class:
                    # colour actual class with yellow
                    ax.set_facecolor('#f7e9c1')

            shap_class_sample = shap_values_in_class[sample_num, :, shap_class]
            shap_class_sample.base_values = base_values[shap_class]
            shap.plots.waterfall(shap_class_sample, show=False)

    fig.set_size_inches(150, 75)
    plt.subplots_adjust(wspace=0.6, hspace=0.3)
    fig.tight_layout()


def shap_beeswarm_bar_pancancer(m: torch.nn.Module,
                                data: torch.Tensor,
                                labels: torch.Tensor = None,
                                e: shap.DeepExplainer = None,
                                model_pred: Literal["positive", "negative", "both"] = "both"):
    pass
    if e is None:
        e = shap.DeepExplainer(m, data)

    shap_values = e(data)
    base_values = e.expected_value

    probs = m(data)
    predictions = torch.argmax(probs, dim=1)

    fig = plt.figure(figsize=(14, 70))
    for prediction_class in range(0, 10):
        if model_pred == "positive":
            class_shap_values = shap_values[predictions == prediction_class, :, prediction_class]
        elif model_pred == "negative":
            class_shap_values = shap_values[predictions != prediction_class, :, prediction_class]
        else:
            class_shap_values = shap_values[:, :, prediction_class]
        class_shap_values.base_values = base_values[prediction_class]

        # plot bee swarm on the left
        ax_swarm = plt.subplot(10, 2, (2 * prediction_class) + 1)
        plt.sca(ax_swarm)
        shap.plots.beeswarm(class_shap_values,
                            show=False)
        # set y label
        ax_swarm.set_ylabel(PAN_CANCER_LABELS[prediction_class], fontsize=48)

        # plot bar on the right
        ax_bar = plt.subplot(10, 2, (2 * prediction_class) + 2)
        plt.sca(ax_bar)
        shap.plots.bar(class_shap_values,
                       ax=ax_bar,
                       show=False)

    fig.set_size_inches(20, 70)
    fig.tight_layout()
    plt.subplots_adjust(top=0.96)

    # set title
    if model_pred == "positive":
        fig.suptitle("SHAP for Positive Predictions", fontsize=64, y=0.98)
    elif model_pred == "negative":
        fig.suptitle("SHAP for Negative Predictions", fontsize=64, y=0.98)
    elif model_pred == "both":
        fig.suptitle("SHAP for All Predictions", fontsize=64, y=0.98)


if __name__ == '__main__':
    from train_pancancer import train_pan_cancer, DEFAULT_DATA_DIR
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    model = train_pan_cancer(pan_cancer_model=model, epochs=30)

    # load data
    samples, slides, data, labels = get_pancancer_data_from_csv(DEFAULT_DATA_DIR)
    data_shap, samples_shap, slides_shap, labels_shap = sklearn.utils.resample(data, samples, slides, labels,
                                                                               stratify=labels,
                                                                               n_samples=1000,
                                                                               replace=False)
    # put in tensor
    data_tensor = torch.FloatTensor(data_shap)
    labels_tensor = torch.LongTensor(labels_shap)

    explainer = shap.DeepExplainer(model, data_tensor)

    # plot
    shap_waterfall_pancancer(model, data_tensor, labels_tensor,
                             e=explainer,
                             sample_names=samples_shap,
                             slide_names=slides_shap,
                             correct="true")
    plt.show()
    plt.close()

    # plot again
    # shap_beeswarm_bar_pancancer(model, data_tensor, labels_tensor,
    #                             e=explainer,
    #                             model_pred = "positive")
    #
    # plt.show()
    # plt.close()
    #
    # shap_beeswarm_bar_pancancer(model, data_tensor, labels_tensor,
    #                             e=explainer,
    #                             model_pred="both")
    # plt.show()
    # plt.close()

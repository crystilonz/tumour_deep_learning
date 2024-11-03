import shap
from pathlib import Path
from models.pan_cancer_classifier import PanCancerClassifier
from train_pancancer import train_pan_cancer, DEFAULT_DATA_DIR
from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_LABELS
from sklearn.model_selection import train_test_split
import torch
from matplotlib import pyplot as plt
from random import randrange


def shap_waterfall_pancancer(m: torch.nn.Module,
                             data: torch.Tensor,
                             label: torch.Tensor,
                             sample_names = None,
                             slide_names = None):

    e = shap.DeepExplainer(m, data)
    shap_values = e(data)
    base_values = e.expected_value

    probs = m(data)
    predictions = torch.argmax(probs, dim=1)

    # we only want to get the correct predictions here
    shap_values_correct = shap_values[label == predictions]
    correct_predictions = predictions[label == predictions]
    correct_samples = sample_names[(label == predictions).tolist()] if sample_names is not None else None
    correct_slides = slide_names[(label == predictions).tolist()] if slide_names is not None else None

    fig = plt.figure(figsize=(100, 50))

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
        shap_values_in_class = shap_values_correct[correct_predictions == sample_class]

        # random sample
        sample_num = randrange(len(shap_values_in_class))
        # get the sample name
        if correct_slides is not None and correct_samples is not None:
            sample_name = correct_samples[sample_num].decode('utf-8')
            slide_name = correct_slides[sample_num].decode('utf-8')
        else:
            sample_name = f"Sample of class {PAN_CANCER_LABELS[sample_class]}"
            slide_name = ""

        # add plot
        col_name = sample_name + '\n' + slide_name
        plt.subplot(1, 10, sample_class + 1)
        plt.title(col_name, fontsize=48, y=1.01)
        plt.axis('off')



        for shap_class in range(0, 10):
            ax = plt.subplot(10, 10, shap_class * 10 + sample_class + 1)
            plt.sca(ax)
            # if diagonal
            if shap_class == sample_class:
                ax.set_facecolor('#ffffe0')
            shap_class_sample = shap_values_in_class[sample_num, :, shap_class]
            shap_class_sample.base_values = base_values[shap_class]
            shap.plots.waterfall(shap_class_sample, show=False)

    fig.set_size_inches(200, 75)
    plt.subplots_adjust(wspace=0.6, hspace=0.3)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)
    model = train_pan_cancer(pan_cancer_model=model, epochs=30)

    # load data
    samples, slides, data, labels = get_pancancer_data_from_csv(DEFAULT_DATA_DIR)
    training_data, testing_data, training_labels, testing_labels = train_test_split(data,
                                                                                    labels,
                                                                                    test_size=0.2,
                                                                                    shuffle=True)
    # put in tensor
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)

    # pass to explainer
    shap_waterfall_pancancer(model, data_tensor[:1100], labels_tensor[:1100], sample_names=samples[:1100], slide_names=slides[:1100])





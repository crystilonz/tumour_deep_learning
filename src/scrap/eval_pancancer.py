import torch
import torch.nn as nn

from data_manipulation.pancancer_from_csv import get_pancancer_data_from_csv, PAN_CANCER_DICT
from models.pan_cancer_classifier import PanCancerClassifier
from train_pancancer import DEFAULT_SAVED_MODEL_DIR, DEFAULT_SAVED_MODEL_NAME, train_pan_cancer, DEFAULT_DATA_DIR
from utils.multiclass_evaluate import accuracy, auroc, roc
from utils.plotting import plot_roc


def eval_panCancer():
    pan_cancer_model = PanCancerClassifier(input_size=33,
                                           hidden_size=42,
                                           output_size=10)
    try:
        pan_cancer_model.load_state_dict(torch.load(DEFAULT_SAVED_MODEL_DIR / DEFAULT_SAVED_MODEL_NAME, weights_only=True))
    except FileNotFoundError:
        # file not found --> retrain
        pan_cancer_model = train_pan_cancer()

    # device agnostic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    samples, slides, data, labels = get_pancancer_data_from_csv(DEFAULT_DATA_DIR)

    # move model to device
    pan_cancer_model.to(device)

    # change to tensor
    data_tensor = torch.tensor(data, dtype=torch.float, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    acc = accuracy(model=pan_cancer_model,
                   data=data_tensor,
                   truth=labels_tensor,
                   classes=10)

    area = auroc(model=pan_cancer_model,
                data=data_tensor,
                truth=labels_tensor,
                classes=10)
    fpr, tpr, threshold = roc(model=pan_cancer_model,
                          data=data_tensor,
                          truth=labels_tensor,
                          classes=10)


    print(f"Model Accuracy: {acc * 100:.02f}%")
    print(f"Area Under ROC: {area:.05f}")

    plot_roc(fpr, tpr, threshold, area,
             label_dict=PAN_CANCER_DICT,
             save_to=DEFAULT_SAVED_MODEL_DIR / "roc_curves")


if __name__ == '__main__':
    eval_panCancer()







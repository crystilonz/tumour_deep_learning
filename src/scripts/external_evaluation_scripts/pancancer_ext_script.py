from utils.external_evaluate import validate_with_external_set
from models.pan_cancer_classifier import PanCancerClassifier

if __name__ == '__main__':
    model = PanCancerClassifier(input_size=34,
                                hidden_size=42,
                                output_size=10)

    validate_with_external_set(model=model,
                               num_classes=7,
                               checkpoint='/Users/muang/PycharmProjects/tumour_deep_learning/src/validation_models/pancancer_classifier/pancancer_classifier_checkpoints.pt')
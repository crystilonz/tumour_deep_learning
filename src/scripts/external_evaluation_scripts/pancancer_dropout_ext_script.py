from utils.external_evaluate import validate_with_external_set
from models.pan_cancer_classifier_with_dropout import PanCancerClassifierWithDropout

if __name__ == '__main__':
    model = PanCancerClassifierWithDropout(input_size=34,
                                           hidden_size=42,
                                           output_size=10)

    validate_with_external_set(model=model,
                               num_classes=7,
                               checkpoint='/Users/muang/PycharmProjects/tumour_deep_learning/src/validation_models/pancancer_dropout_model/pancancer_dropout_checkpoints.pt')

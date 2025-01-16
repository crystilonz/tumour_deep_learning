from utils.external_evaluate import validate_with_external_set
from models.pan_cancer_classifier_with_dropout import PanCancerClassifierWithDropout
from script_info import GTEX_details

if __name__ == '__main__':
    model = PanCancerClassifierWithDropout(input_size=34,
                                           hidden_size=42,
                                           output_size=10)

    validate_with_external_set(model=model,
                               num_classes=GTEX_details.num_classes,
                               checkpoint='/Users/muang/PycharmProjects/tumour_deep_learning/src/validation_models/pancancer_dropout_model/pancancer_dropout_checkpoints.pt',
                               external_csv=GTEX_details.ext_directory,
                               output_dir=GTEX_details.output_directory)

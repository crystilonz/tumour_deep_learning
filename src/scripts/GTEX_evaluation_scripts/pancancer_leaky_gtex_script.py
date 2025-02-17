from utils.external_evaluate import validate_with_external_set
from models.pan_cancer_leaky import LeakyPanCancerClassifier
from script_info import GTEX_details

if __name__ == '__main__':
    model = LeakyPanCancerClassifier(input_size=34,
                                     hidden_size=42,
                                     output_size=10)

    validate_with_external_set(model=model,
                               num_classes=GTEX_details.num_classes,
                               checkpoint='/Users/muang/PycharmProjects/tumour_deep_learning/src/validation_models/leaky_pancancer/leaky_pancancer_checkpoints.pt',
                               external_csv=GTEX_details.ext_directory,
                               output_dir=GTEX_details.output_directory,
                               confusion_mode='gtex')

from utils.external_evaluate import validate_with_external_set
from models.logistic_regression import PanCancerLogisticRegression

if __name__ == '__main__':
    model = PanCancerLogisticRegression(input_size=34,
                                        output_size=10)

    validate_with_external_set(model=model,
                               num_classes=7,
                               checkpoint='/Users/muang/PycharmProjects/tumour_deep_learning/src/validation_models/logistic_regression/logistic_regression_checkpoints.pt')
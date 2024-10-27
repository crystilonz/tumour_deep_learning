from k_fold_validation_pancancer import k_fold_validation_pancancer
from models.logistic_regression import PanCancerLogisticRegression

if __name__ == '__main__':
    model_lambda = lambda: PanCancerLogisticRegression(input_size=34,
                                        output_size=10)
    k_fold_validation_pancancer(model_lambda,
                                epochs=30,
                                results_dir="logistic_regression",
                                best_model_save_name="logistic_regression_checkpoints")
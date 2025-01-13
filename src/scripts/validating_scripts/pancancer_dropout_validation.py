from k_fold_validation_pancancer import k_fold_validation_pancancer
from models.pan_cancer_classifier_with_dropout import PanCancerClassifierWithDropout

if __name__ == '__main__':
    model_lambda = lambda: PanCancerClassifierWithDropout(input_size=34,
                                                          hidden_size=42,
                                                          output_size=10)
    k_fold_validation_pancancer(model_lambda,
                                results_dir="pancancer_dropout_model",
                                best_model_save_name="pancancer_dropout_checkpoints")

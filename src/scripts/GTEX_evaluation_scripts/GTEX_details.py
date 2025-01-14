from pathlib import Path

ext_directory = Path(__file__).parent.parent.parent / 'datasets' / 'gtex_pancancer'
output_directory = Path(__file__).parent.parent.parent / 'gtex_validation_results'
num_classes = 9
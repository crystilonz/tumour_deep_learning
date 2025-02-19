import argparse
import torch
from utils.pytorch_run_FSDP import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=8,
                        metavar='N', help='Number of GPUs')
    parser.add_argument('--pkl', type=str, default='./datasets/lung_text/splits.pkl',
                        metavar='pickle/file/path', help='path to pickle file containing dataset splits')
    parser.add_argument('--vocab_dir', type=str, default='./datasets/lung_text/vocab.json',
                        metavar='vocab/file/path', help='path to vocab file')

    parser.add_argument('--train_batch_size', type=int, default=32,
                        metavar='N', help='batch size for training')
    parser.add_argument('--validate_batch_size', type=int, default=32,
                        metavar='N', help='batch size for validation')
    parser.add_argument('--num_workers', type=int, default=8,
                        metavar='N', help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-5,
                        metavar='LR', help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        metavar='N', help='number of epochs')

    # model args
    parser.add_argument('--input_size', type=int, default=128,
                        metavar='N', help='input size of model')
    parser.add_argument('--hidden_size', type=int, default=128,
                        metavar='N', help='hidden size of model')
    parser.add_argument('--embed_size', type=int, default=128,
                        metavar='N', help='embedding size of model')
    parser.add_argument('--num_layers', type=int, default=1,
                        metavar='N', help='number of layers of LSTM')

    # flags
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='whether to save model')
    
    # save directory
    parser.add_argument('--save_dir', type=str, default='./dist_models/LSTM',
                        metavar='path/to/save/directory', help='where to save the results')

    args = parser.parse_args()
    fsdp_main(args)
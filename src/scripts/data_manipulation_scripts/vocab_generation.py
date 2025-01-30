from data_manipulation.lung_caption_vocab import extract_text_list_from_consensus, Vocabulary
from pathlib import Path


consensus_csv_path = Path(__file__).parent.parent.parent / 'datasets/lung_text/consensus.csv'
vocab_save_path = Path(__file__).parent.parent.parent / 'datasets/lung_text/vocab.json'

# vocab generation settings
DEFAULT_MAX_VOCAB = 1024
DEFAULT_MIN_FREQUENCY = 1


if __name__ == "__main__":
    consensus_list = extract_text_list_from_consensus(consensus_csv_path)
    vocab = Vocabulary(max_size=DEFAULT_MAX_VOCAB, min_freq=DEFAULT_MIN_FREQUENCY)
    vocab.build_vocab(consensus_list)
    vocab.save(vocab_save_path)

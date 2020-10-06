import argparse
import os
import shutil

from pathlib import Path
from tokenizers import BertWordPieceTokenizer


def main():
    parser = argparse.ArgumentParser('Parse a file')

    parser.add_argument('--vocab_size', '-s', help='size of vocab',
                        type=int, required=False, default=50_000)

    parser.add_argument('--txt_folder', '-t', help='Path to folder with txt files',
                        type=str, required=False, default="input")

    parser.add_argument('--save_folder', '-o', help='Path to output',
                        type=str, required=False, default="data")

    args = parser.parse_args()
    vocab_size = args.vocab_size
    input_folder = args.txt_folder
    save_path = args.save_folder

    # Initialize a tokenizer
    tokenizer = BertWordPieceTokenizer(clean_text=True,
                                       handle_chinese_chars=False,
                                       strip_accents=False,
                                       lowercase=True,)

    txt_paths = [str(x) for x in Path(input_folder).glob("**/*.txt")]

    # Customize training
    tokenizer.train(files=txt_paths,
                    vocab_size=vocab_size,
                    min_frequency=2,
                    special_tokens=['[PAD]', '[UNK]',
                                    '[CLS]', '[SEP]', '[MASK]'],
                    wordpieces_prefix="##",
                    )

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    tokenizer.save_model(save_path)


if __name__ == "__main__":
    main()

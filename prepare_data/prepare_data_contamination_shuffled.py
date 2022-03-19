import argparse
import random

def main():
    contamination_copies = int(args.contamination_copies)
    wiki_size = int(float(args.wiki_size))
    random.seed(42)

    # get wiki data
    with open(arge.wiki_path, 'r') as f:
        wiki_text = [next(f) for _ in range(wiki_size)]
        wiki_text = list(map(lambda s: s.strip(), wiki_text))  # remove '\n' from end of instance

    # get contaminated data
    with open(args.contamination_path, 'r') as f:
        contaminted_text = f.read().splitlines()
        contaminted_text = contaminted_text * contamination_copies

    # combine
    text = wiki_text + contaminted_text
    random.shuffle(text)

    with open(f'{args.path}wiki_{args.wiki_size}_contamination_copies_{contamination_copies}.txt', 'w') as f:
        f.write('\n'.join(text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--contamination_path", help="path to contaminated txt file")
    parser.add_argument("--contamination_copies", default=100, help="number of contaminated copies (default=100).")
    parser.add_argument("--wiki_path", help="path to Wikipedia txt file")
    parser.add_argument("--wiki_size", default=1e6, help="size of Wikipedia in lines (default=1e6)")
    parser.add_argument("--path", help="path to the new file. i.e, data_folder/")
    args = parser.parse_args()
    main()




    





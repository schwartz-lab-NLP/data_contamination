import argparse
import random


def main():
    contamination_copies = 100
    wiki_size = int(1e6)
    random.seed(42)

    # get wiki data
    with open(arge.wiki_path, 'r') as f:
        wiki_text = [next(f) for _ in range(wiki_size)]
        wiki_text = list(map(lambda s: s.strip(), wiki_text))  # remove '\n' from end of instance

    # get contaminated data
    with open(args.contamination_path, 'r') as f:
        contaminted_text = f.read().splitlines()
        contaminted_text = contaminted_text * contamination_copies

    first_wiki_size = 400000
    middle_wiki_size = 400000
    # last_wiki_size = 200000 # this variable is not truly required

    first_text = wiki_text[:first_wiki_size]
    random.shuffle(first_text)
    middle_text = wiki_text[first_wiki_size:first_wiki_size + middle_wiki_size]
    random.shuffle(middle_text)
    last_text = wiki_text[first_wiki_size + middle_wiki_size:] + contaminted_text
    random.shuffle(last_text)

    # combine
    text = first_text + middle_text + last_text

    with open(f'{args.path}wiki_{args.wiki_size}_contamination_copies_{sst_size}_last.txt', 'w') as f:
        f.write('\n'.join(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--contamination_path", help="path to contaminated txt file")
    parser.add_argument("--wiki_path", help="path to Wikipedia txt file")
    parser.add_argument("--path", help="path to the new file. i.e, data_folder/")
    args = parser.parse_args()
    main()










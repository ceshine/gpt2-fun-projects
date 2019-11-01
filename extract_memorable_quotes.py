from tqdm import tqdm

INPUT_PATH = "data/cornell_movie_quotes/moviequotes.memorable_quotes.txt"
OUTPUT_PATH = "data/cornell_movie_quotes/memorable_quotes.txt"


def main():
    with open(INPUT_PATH, "r") as fin:
        lines = []
        for i, line in tqdm(enumerate(fin.readlines())):
            if i % 4 in (2, 3):
                continue
            lines.append(line)
    with open(OUTPUT_PATH, "w") as fout:
        fout.writelines(lines)


if __name__ == "__main__":
    main()

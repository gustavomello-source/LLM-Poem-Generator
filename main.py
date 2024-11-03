import argparse
import os

from torch.utils.data import DataLoader
from datasets import load_dataset

# Local imports
from writer import Writer
from fetcher import PoetrydbFetcher
from dataset import PoemDataset
from Poet import PoetryModel
from fetcher import convert_json_to_txt

def main():
    parser = argparse.ArgumentParser(description='The script executes the pipeline to fetch poetry data from the websites and generate poems using llama 3.1-8B model.') 
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--fetch', action='store_true', help='Fetch data from Poetrydb.')

    args = parser.parse_args()

    if os.path.isdir('./Poetrydb') and args.fetch:
        print(f"Poetrydb directory already exists.\nDo you want to continue? [y/n]")
        answer = input()
        if answer.lower() != 'y':
            print("\nNot Fetching data from Poetrydb...")

            if not os.path.isfile('./processed_poems.txt'):
                print("\nProcessed poems file not found.")

            print("Trying to find json files...")
            try:
                convert_json_to_txt('./Poetrydb')
            except Exception as e:
                print(e)
                print("\nJson files not found. Exiting...")
                return
        else:
            PoetrydbFetcher(url="https://poetrydb.org/", verbose=args.verbose)

    elif args.fetch:
        PoetrydbFetcher(url="https://poetrydb.org/", verbose=args.verbose)


    # Instantiate the PoetGenerator class
    poet = PoetryModel()

    # Load the dataset
    if not os.path.isfile('./processed_poems.txt'):
        print("\nProcessed poems file not found.")

        return
    
    poems = load_dataset("text", data_files="processed_poems.txt")["train"]["text"]
    dataset = PoemDataset(poems, poet.tokenizer)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    poet.fine_tune(dataloader, num_epochs=3, learning_rate=5e-5)

    example_new_poem  = poet.generate_poem("Write a poem about crows, ravens and england in classic style.")
    print(example_new_poem)

    poet.save_model("./fine_tuned_poetry_model")

if __name__ == '__main__':
    main()
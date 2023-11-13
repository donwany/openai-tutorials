import pandas as pd
from datasets import load_dataset


def load_and_process_dataset(dataset_name, output_file):
    # Load a dataset by name
    dataset = load_dataset(dataset_name, split="train")

    # Create a DataFrame and select relevant columns
    df = pd.DataFrame(dataset).reset_index(drop=True)
    df = df[['request', 'response-1']]

    # Convert DataFrame to JSON records
    json_data = df.to_json(orient="records")

    # Save JSON data to a file
    with open(output_file, 'w') as file:
        file.write(json_data)


if __name__ == '__main__':
    dataset_name = "argilla/llama-2-banking-fine-tune"
    output_file = 'bank_dataset.json'

    load_and_process_dataset(dataset_name, output_file)

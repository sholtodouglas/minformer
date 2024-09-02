import os
import urllib.request
import data  # Assuming this is your custom module with CharDataset

def download_file(url, filename):
    """Download a file if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists. Skipping download.")

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a 'data' directory in the same location as the script
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    input_file_path = os.path.join(data_dir, 'TinyStoriesV2-GPT4-train.txt')
    download_file(url, input_file_path)

    # Create CharDataset instance
    ds = data.CharDataset(data.CharDataset.get_default_config())

    # Create packed records
    output_dir = os.path.join(data_dir, 'tfrecords/')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating packed records at {output_dir}...")
    ds.create_packed_records(
        input_file_path=input_file_path,
        output_dir=output_dir,
        custom_delimiter='<|endoftext|>'
    )
    print("Packed records created successfully.")

if __name__ == "__main__":
    main()
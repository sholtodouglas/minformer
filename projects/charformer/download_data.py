"""Downloads data for charformer.


python3 projects/charformer/download_data.py

python3 projects/charformer/download_data.py --use_gcs --bucket_name minformer_data --gcs_output_dir charformer/tiny_stories/tfrecords/
"""

import os
import urllib.request
import data  # Assuming this is your custom module with CharDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Download and process data for Charformer")
    parser.add_argument("--use_gcs", action="store_true", help="Use Google Cloud Storage")
    parser.add_argument("--bucket_name", type=str, help="GCS bucket name")
    parser.add_argument("--gcs_output_dir", type=str, default="charformer/tiny_stories/tfrecords/", help="GCS output directory")
    return parser.parse_args()

def download_file(url, filename):
    """Download a file if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists. Skipping download.")

def main():
    args = parse_args()
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

    # Set up output directory
    if args.use_gcs:
        output_dir = f"gs://{args.bucket_name}/{args.gcs_output_dir}"
        print(f"Creating packed records in GCS bucket {args.bucket_name} at {output_dir}...")
    else:
        output_dir = os.path.join(data_dir, 'tfrecords/')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating packed records at {output_dir}...")
    
    print(f"Creating packed records at {output_dir}...")
    ds.create_packed_records(
        input_file_path=input_file_path,
        output_dir=output_dir,
        custom_delimiter='<|endoftext|>'
    )
    print("Packed records created successfully.")

if __name__ == "__main__":
    main()
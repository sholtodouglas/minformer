"""Downloads data for charformer.


python3 projects/bio/download_data.py

python3 projects/bio/download_data.py --dataset open-genome-imgpr --use-gcs --bucket-name minformer_data --sequence-length=16384
"""

import argparse
import os
import urllib.request

import data
import data_hf
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download and process DNA data")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["human-genome-8192", "open-genome-imgpr"],
        default="open-genome-imgpr",
        help="Type of dataset to download and process",
    )
    parser.add_argument("--use-gcs", action="store_true", help="Use Google Cloud Storage")
    parser.add_argument("--bucket-name", type=str, help="GCS bucket name")
    parser.add_argument("--sequence-length", type=int, default=8192, help="Training seqlen")
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
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Set up output directory
    if args.use_gcs:
        output_dir = f"gs://{args.bucket_name}/{args.dataset}/tfrecords/"
    else:
        output_dir = os.path.join(data_dir, f"{args.dataset}/tfrecords/")
        os.makedirs(output_dir, exist_ok=True)

    print(f"Creating packed records at {output_dir}...")

    # Download and process the dataset based on the selected dataset type
    if args.dataset == "human-genome-8192":
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz"
        )
        input_file_path = os.path.join(data_dir, "human_genome.fa.gz")
        download_file(url, input_file_path)
        ds = data.DNADataset(sequence_length=args.sequence_length)
        ds.create_tfrecords(
            input_file_path=input_file_path,
            output_dir=output_dir,
        )
    elif args.dataset == "open-genome-imgpr":
        # https://huggingface.co/datasets/LongSafari/open-genome/resolve/main/stage1/gtdb/gtdb_train_shard_0.parquet
        data_files = {
            "train": "stage1/imgpr/imgpr_train.parquet",
            "test": "stage1/imgpr/imgpr_test.parquet",
        }
        # save as parquet using dataset builder
        hf_ds = load_dataset(
            "LongSafari/open-genome", name="stage1", cache_dir=data_dir, data_files=data_files, num_proc=8
        )
        data_hf.process_and_save_tfrecords(
            hf_ds["train"],
            os.path.join(output_dir, "stage1/train_v2"),
            sequence_length=args.sequence_length,
        )
        # data_hf.process_and_save_tfrecords(hf_ds['test'], os.path.join(output_dir, "stage1/test"), sequence_length=16384)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print("Packed records created successfully.")


if __name__ == "__main__":
    main()

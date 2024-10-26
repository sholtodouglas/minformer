import os
import re
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import pandas as pd

PADDING = "P"
VOCAB = [PADDING, "U", "A", "C", "G", "T", "N"]  #  padding, unknown, A, C, G, T
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
tokenize = lambda x: [stoi.get(ch, 0) for ch in x]
detokenize = lambda x: "".join([itos.get(i, "U") for i in x])

LAD_CAT = {
    "inter-LAD": 0,
    "LAD": 1,
    "LAD boundary": 2,
}
LAD_CAT_REV = {v:k for k,v in LAD_CAT.items()}

SAD_CAT = {
    "inter-SAD": 0,
    "SAD": 1,
    "SAD boundary": 2,
}
SAD_CAT_REV = {v:k for k,v in SAD_CAT.items()}

CHROM = {
    "chr1": 0,
    "chr2": 1,
    "chr3": 2,
    "chr4": 3,
    "chr5": 4,
    "chr6": 5,
    "chr7": 6,
    "chr8": 7,
    "chr9": 8,
    "chr10": 9,
    "chr11": 10,
    "chr12": 11,
    "chr13": 12,
    "chr14": 13,
    "chr15": 14,
    "chr16": 15,
    "chr17": 16,
    "chr18": 17,
    "chr19": 18,
    "chr20": 19,
    "chr21": 20,
    "chr22": 21,
    "chrX": 22,
    "chrY": 23,
}
CHROM_REV = {v:k for k,v in CHROM.items()}

CELL_TYPE = {
    "intermediate_progenitor": 0,
    "excitatory_neuron": 1,
    "radial_glia": 2,
    "unknown": 3
}
CELL_REV = {v:k for k,v in CELL_TYPE.items()}

def next_multiple(x, n):
    return x + (-x % n)


def process_dfs(ip_df_base, en_df_base, rg_df_base, sequence_length):
    ip_df_base['cell_type'] = 'intermediate_progenitor'
    en_df_base['cell_type'] = 'excitatory_neuron'
    rg_df_base['cell_type'] = 'radial_glia'
    # def add_conservation_label(ip_df, en_df, rg_df):
    lad_conserved = (ip_df_base['LMNB1 Cat'] == en_df_base['LMNB1 Cat']) & (ip_df_base['LMNB1 Cat'] == rg_df_base['LMNB1 Cat'])
    ip_df_base['lad_conserved'] = lad_conserved
    en_df_base['lad_conserved'] = lad_conserved
    rg_df_base['lad_conserved'] = lad_conserved

    sad_conserved = (ip_df_base['SON Cat'] == en_df_base['SON Cat']) & (ip_df_base['SON Cat'] == rg_df_base['SON Cat'])
    ip_df_base['sad_conserved'] = sad_conserved
    en_df_base['sad_conserved'] = sad_conserved
    rg_df_base['sad_conserved'] = sad_conserved

    # First, separate out chr19 data before shuffling
    chr19_ip = ip_df_base[ip_df_base['Chrom'] == 'chr19'].copy()
    chr19_en = en_df_base[en_df_base['Chrom'] == 'chr19'].copy()
    chr19_rg = rg_df_base[rg_df_base['Chrom'] == 'chr19'].copy()

    # Remove chr19 from main dataframes
    ip_df = ip_df_base[ip_df_base['Chrom'] != 'chr19'].copy()
    en_df = en_df_base[en_df_base['Chrom'] != 'chr19'].copy()
    rg_df = rg_df_base[rg_df_base['Chrom'] != 'chr19'].copy()

    ip_df = ip_df.sample(frac=1, random_state=42).reset_index(drop=True)
    en_df = en_df.sample(frac=1, random_state=42).reset_index(drop=True)
    rg_df = rg_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Pull out anything where Chrom == chr19, don't include it in train or val.

    # Get the training sets (first 80% of rows)
    train_ip = ip_df.head(int(len(ip_df) * 0.8))
    train_en = en_df.head(int(len(en_df) * 0.8))
    train_rg = rg_df.head(int(len(rg_df) * 0.8))

    # Get the validation sets (last 20% of rows), we need this to be consistent so 
    # we don't train on the val set of another cell.
    val_ip = ip_df.tail(int(len(ip_df) * 0.2))
    val_en = en_df.tail(int(len(en_df) * 0.2))
    val_rg = rg_df.tail(int(len(rg_df) * 0.2))

    # Create chr19 validation set
    chr19_combined = pd.concat([chr19_ip, chr19_en, chr19_rg], ignore_index=True)
    chr19_combined = chr19_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Combine into single training and validation sets
    train_combined = pd.concat([train_ip, train_en, train_rg], ignore_index=True)
    train_combined = train_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    val_combined = pd.concat([val_ip, val_en, val_rg], ignore_index=True)
    val_combined = val_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    process_rows(chr19_combined, output_dir=f"gs://minformer_data/lab_data_chr19/tfrecords/", bucket=sequence_length)
    process_rows(val_combined, output_dir=f"gs://minformer_data/lab_data_val/tfrecords/", bucket=sequence_length)
    process_rows(train_combined, output_dir=f"gs://minformer_data/lab_data_train/tfrecords/", bucket=sequence_length)



def process_rows(df, output_dir, bucket: int):

    df = df.sample(frac=1).reset_index(drop=True)
    record_count = 0
    save_together = 128
    save_together_rows = []

    for i in tqdm(range(0, len(df))):
        row = df.iloc[i]
        if row["Sequence"][0:5] != "NNNNN":
            save_together_rows.append(row)

        if len(save_together_rows) == save_together:
            output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")

            with tf.io.TFRecordWriter(output_file) as writer:
                for row in save_together_rows:

                    tokens = tokenize(row["Sequence"])
                    padding = bucket - len(tokens)
                    segment_ids = np.ones_like(tokens)
                    tokens = np.pad(tokens, (0, padding))
                    segment_ids = np.pad(segment_ids, (0, padding))
                    if "LMNB1 Cat" in row:
                        lad_category = LAD_CAT[row["LMNB1 Cat"]]
                        lad_value = row["LMNB1 Signal"]
                        sad_category = SAD_CAT[row["SON Cat"]]
                        sad_value = row["SON Signal"]
                        chromosome = row["Chrom"]  # Keep chromosome as string
                        # Add conservation information
                        lad_conserved = int(row.get("lad_conserved", False))
                        sad_conserved = int(row.get("sad_conserved", False))
                        cell_type = CELL_TYPE[row.get("cell_type", "unknown")]

                    else:
                        lad_category = 0
                        lad_value = 0
                        sad_category = 0
                        sad_value = 0
                        chromosome = "NA"
                        # Add conservation information
                        lad_conserved = int(row.get("lad_conserved", False))
                        sad_conserved = int(row.get("sad_conserved", False))
                        cell_type = CELL_TYPE[row.get("cell_type", "unknown")]

                    save_tfrecord(
                        writer, tokens, segment_ids, lad_category, lad_value,
                        sad_category, sad_value, chromosome, lad_conserved,
                        sad_conserved, cell_type
                    )

            record_count += 1
            save_together_rows = []
        else:
            pass


def save_tfrecord(writer, tokens, segment_ids, lad_category, lad_value,
                 sad_category, sad_value, chromosome, lad_conserved,
                 sad_conserved, cell_type):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                "lad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[lad_category])),
                "lad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[lad_value])),
                "sad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[sad_category])),
                "sad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[sad_value])),
                "chromosome": tf.train.Feature(bytes_list=tf.train.BytesList(value=[chromosome.encode()])),
                "lad_conserved": tf.train.Feature(int64_list=tf.train.Int64List(value=[lad_conserved])),
                "sad_conserved": tf.train.Feature(int64_list=tf.train.Int64List(value=[sad_conserved])),
                "cell_type": tf.train.Feature(int64_list=tf.train.Int64List(value=[cell_type]))
            }
        )
    )
    writer.write(example.SerializeToString())


def feature_description() -> Any:
    return {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lad_category": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "lad_value": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "sad_category": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sad_value": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        "chromosome": tf.io.FixedLenFeature([], tf.string),
        "lad_conserved": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "sad_conserved": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "cell_type": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }



def load_and_retokenize_tfrecord(file_path: str):
    retokenized_data = []
    feature_data = {
        "segment_ids": [],
        "lad_category": [],
        "lad_value": [],
        "sad_category": [],
        "sad_value": [],
        "chromosome": [],
        "lad_conserved": [],
        "sad_conserved": [],
        "cell_type": [],
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())

    dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = dataset.map(_parse_function)

    for parsed_record in parsed_dataset:
        x = parsed_record["x"].numpy()

        original_sequence = detokenize(x)  # Assuming detokenize function is defined elsewhere
        retokenized_data.append(original_sequence)

        for feature in feature_data.keys():
            if feature == "chromosome":
                feature_data[feature].append(parsed_record[feature].numpy().decode())  # Decode bytes to string
            else:
                feature_data[feature].append(parsed_record[feature].numpy())

    return retokenized_data, feature_data


def create_iterator(stage_1: list[str], stage_2: list[str], batch_size: int, shuffle: bool = False):
    """Creates a python iterator to load batches."""

    def _parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description())
        return parsed_features

    # List all files matching the patterns
    stage_1_files = []
    for pattern in stage_1:
        stage_1_files.extend(tf.io.gfile.glob(pattern))

    # Shuffle the file list
    random.shuffle(stage_1_files)

    print(f"Found {len(stage_1_files)} files for stage 1")

    # Now (for example human genome), we want to have the end of
    # training focused on this.
    stage_2_files = []
    for pattern in stage_2:
        stage_2_files.extend(tf.io.gfile.glob(pattern))

    print(f"Found {len(stage_2_files)} files for stage 2")

    # Shuffle the file list
    random.shuffle(stage_2_files)

    # Combine them.
    files = stage_1_files + stage_2_files
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function, num_parallel_calls=32)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(32)

    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            "lad_category": batch["lad_category"].numpy().astype(np.int32),
            "lad_value": batch["lad_value"].numpy().astype(np.float32),
            "sad_category": batch["sad_category"].numpy().astype(np.int32),
            "sad_value": batch["sad_value"].numpy().astype(np.float32),
            "chromosome": np.array([batch["chromosome"].numpy()]),
            "lad_conserved": batch["lad_conserved"].numpy().astype(np.int32),
            "sad_conserved": batch["sad_conserved"].numpy().astype(np.int32),
            "cell_type": batch["cell_type"].numpy().astype(np.int32)
        }
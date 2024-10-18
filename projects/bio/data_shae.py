import os
import re
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

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

SAD_CAT = {
    "inter-SAD": 0,
    "SAD": 1,
    "SAD boundary": 2,
}

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


def next_multiple(x, n):
    return x + (-x % n)


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
                    else:
                        lad_category = 0
                        lad_value = 0
                        sad_category = 0
                        sad_value = 0
                        chromosome = "NA"
                    save_tfrecord(
                        writer, tokens, segment_ids, lad_category, lad_value, sad_category, sad_value, chromosome
                    )

            record_count += 1
            save_together_rows = []
        else:
            pass


def save_tfrecord(writer, tokens, segment_ids, lad_category, lad_value, sad_category, sad_value, chromosome):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                "lad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[lad_category])),
                "lad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[lad_value])),
                "sad_category": tf.train.Feature(int64_list=tf.train.Int64List(value=[sad_category])),
                "sad_value": tf.train.Feature(float_list=tf.train.FloatList(value=[sad_value])),
                "chromosome": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[chromosome.encode()])
                ),  # Save as bytes
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
        "chromosome": tf.io.FixedLenFeature([], tf.string),  # Change to string
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
            "chromosome": np.array([batch["chromosome"].numpy()]),  # Decode bytes to string
        }

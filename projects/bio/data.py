import gzip
import os
import re
from typing import Any, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class DNADataset:
    def __init__(self, sequence_length: int = 8192):
        self.sequence_length = sequence_length
        self.bases = ["A", "C", "G", "T"]
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.bases)}
        self.stoi["<unk>"] = 0
        self.itos = {i + 1: ch for i, ch in enumerate(self.bases)}
        self.itos[0] = "<unk>"

    @property
    def vocab_size(self):
        return len(self.bases) + 1  # +1 for <unk>

    def tokenize(self, sequence):
        return np.array([self.stoi.get(base, 0) for base in sequence], dtype=np.int32)

    def detokenize(self, tokens):
        return "".join([self.itos.get(token, "<unk>") for token in tokens])

    @property
    def feature_description(self) -> Any:
        return {
            "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        }

    def preprocess_dna_sequence(self, sequence):
        sequence = re.sub(r"[^ACGT]", "", sequence.upper())
        return [sequence[i : i + self.sequence_length] for i in range(0, len(sequence), self.sequence_length)]

    def create_tfrecords(self, input_file_path: str, output_dir: str):
        def save_tfrecord(writer, tokens, segment_ids):
            x = tokens
            segment_ids = segment_ids
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                        "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
                    }
                )
            )
            writer.write(example.SerializeToString())

        os.makedirs(output_dir, exist_ok=True)
        files_saved_so_far = 0
        with gzip.open(input_file_path, "rt") as file:
            fasta_data = file.read()
        fasta_records = fasta_data.split(">")[1:]

        for record in tqdm(fasta_records, desc="Processing FASTA records"):
            if record.strip():
                lines = record.strip().split("\n")
                sequence = "".join(lines[1:])
                preprocessed_chunks = self.preprocess_dna_sequence(sequence)

                for sequence in tqdm(preprocessed_chunks, desc="Processing single example"):
                    new_output_file = os.path.join(output_dir, f"record_{files_saved_so_far}.tfrecord")
                    if len(sequence) == self.sequence_length:
                        with tf.io.TFRecordWriter(new_output_file) as writer:
                            tokens = self.tokenize(sequence)
                            save_tfrecord(writer, tokens, np.ones_like(tokens))
                        files_saved_so_far += 1
                    else:
                        print(f"Skipping sequence. Too short, length {len(sequence)}")

    def load_and_retokenize_tfrecord(self, file_path: str) -> List[str]:
        """
        Loads a TFRecord file and retokenizes its content according to the current DNADataset instance.

        Args:
        file_path (str): Path to the TFRecord file.

        Returns:
        List[str]: A list of retokenized DNA sequences.
        """
        retokenized_data = []

        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, self.feature_description)

        dataset = tf.data.TFRecordDataset(file_path)
        parsed_dataset = dataset.map(_parse_function)

        for parsed_record in parsed_dataset:
            x = parsed_record["x"].numpy()
            original_sequence = self.detokenize(x)
            retokenized_data.append(original_sequence)

        return retokenized_data

    def create_iterator(self, file_pattern: str, batch_size: int, shuffle: bool = False):
        """Creates a python iterator to load batches."""

        def _parse_function(example_proto):
            parsed_features = tf.io.parse_single_example(example_proto, self.feature_description)
            return parsed_features

        files = tf.data.Dataset.list_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        for batch in dataset:
            yield {
                "x": batch["x"].numpy().astype(np.int32),
                "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
            }

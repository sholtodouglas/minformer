import os
import re
from typing import Any

import numpy as np
import tensorflow as tf
from tqdm import tqdm

PADDING = "P"
VOCAB = [PADDING, "U", "A", "C", "G", "T"]  #  padding, unknown, A, C, G, T
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
tokenize = lambda x: [stoi.get(ch, 0) for ch in x]
detokenize = lambda x: "".join([itos.get(i, "U") for i in x])

CAUSAL_CONV_MAX_WINDOW = 32


def preprocess_dna_sequence(x, sequence_length):
    # remove all non ACGT characters and convert to uppercase
    x = re.sub(r"[^ACGT]", "", x.upper())
    # split into chunks of SEQ_LEN, append start/end token.
    return [x[i : i + sequence_length] for i in range(0, len(x), sequence_length)]


def next_multiple(x, n):
    return x + (-x % n)


def process_and_save_tfrecords(dataset, output_dir, sequence_length):
    os.makedirs(output_dir, exist_ok=True)
    current_tokens = []
    current_segment_ids = []
    record_count = 0
    sequence_number = 1

    tokens_to_save = []
    segment_ids_to_save = []

    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        sequence = example["text"]
        chunks = preprocess_dna_sequence(sequence, sequence_length)

        for chunk_idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            original_token_len = len(tokens)
            padded_token_len = next_multiple(original_token_len, CAUSAL_CONV_MAX_WINDOW)
            padding = padded_token_len - original_token_len
            tokens = tokens + [0] * padding
            assert len(tokens) == padded_token_len
            if len(current_tokens) + len(tokens) <= sequence_length:
                current_tokens.extend(tokens)
                # Add new segment id for the non padding tokens.
                current_segment_ids.extend([sequence_number] * original_token_len)
                current_segment_ids.extend([0] * padding)
                sequence_number += 1
            else:
                # Save the current stacked sequences
                if len(current_tokens) > 0:
                    tokens_to_save.append(current_tokens)
                    segment_ids_to_save.append(current_segment_ids)
                    sequence_number = 1

                # Start a new sequence
                current_tokens = tokens
                current_segment_ids = [sequence_number] * original_token_len
                current_segment_ids.extend([0] * padding)

            # Check if we have a full sequence to save so we can save immediately
            if len(current_tokens) == sequence_length:
                tokens_to_save.append(current_tokens)
                segment_ids_to_save.append(current_segment_ids)
                sequence_number = 1
                current_tokens = []
                current_segment_ids = []
        # Save to write every 100 sequences.
        if len(tokens_to_save) > 500:
            save_records(output_dir, record_count, tokens_to_save, segment_ids_to_save, sequence_length)
            tokens_to_save = []
            segment_ids_to_save = []
            record_count += 1

    # Save any remaining sequences
    if len(tokens_to_save) > 0:
        save_records(output_dir, record_count, tokens_to_save, segment_ids_to_save, sequence_length)


def save_tfrecord(writer, tokens, segment_ids):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
                "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
            }
        )
    )
    writer.write(example.SerializeToString())


def save_records(output_dir, record_count, tokens_list, segment_ids_list, sequence_length):
    # Pad if necessary

    output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")
    with tf.io.TFRecordWriter(output_file) as writer:
        for tokens, segment_ids in zip(tokens_list, segment_ids_list):
            if len(tokens) < sequence_length:
                padding_length = sequence_length - len(tokens)
                tokens = np.concatenate([tokens, np.full(padding_length, stoi["P"])])
                segment_ids = segment_ids + [0] * padding_length
            save_tfrecord(writer, tokens, segment_ids)


def feature_description() -> Any:
    return {
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "segment_ids": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }


def load_and_retokenize_tfrecord(file_path: str) -> list[str]:
    """
    Loads a TFRecord file and retokenizes its content according to the current DNADataset instance.

    Args:
    file_path (str): Path to the TFRecord file.

    Returns:
    List[str]: A list of retokenized DNA sequences.
    """
    retokenized_data = []
    segment_ids = []

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description())

    dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = dataset.map(_parse_function)

    for parsed_record in parsed_dataset:
        x = parsed_record["x"].numpy()
        seg_ids = parsed_record["segment_ids"].numpy()
        original_sequence = detokenize(x)
        retokenized_data.append(original_sequence)
        segment_ids.append(seg_ids)

    return retokenized_data, segment_ids


def create_iterator(file_pattern: str, batch_size: int, shuffle: bool = False):
    """Creates a python iterator to load batches."""

    def _parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description())
        return parsed_features

    files = tf.data.Dataset.list_files(file_pattern)
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
        }

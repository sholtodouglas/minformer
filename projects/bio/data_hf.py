import os
import re
from typing import Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm


VOCAB = ["P", "U", "A", "C", "G", "T"]  #  padding, unknown, A, C, G, T
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
tokenize = lambda x: [stoi.get(ch, 0) for ch in x]
detokenize = lambda x: "".join([itos.get(i, "U") for i in x])


def preprocess_dna_sequence(x, sequence_length):
    # remove all non ACGT characters and convert to uppercase
    x = re.sub(r"[^ACGT]", "", x.upper())
    # split into chunks of SEQ_LEN, append start/end token.
    return [x[i : i + sequence_length] for i in range(0, len(x), sequence_length)]


def process_and_save_tfrecords(dataset, output_dir, sequence_length):
    os.makedirs(output_dir, exist_ok=True)
    current_tokens = []
    current_segment_ids = []
    record_count = 0
    sequence_number = 1

    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        sequence = example["text"]
        chunks = preprocess_dna_sequence(sequence, sequence_length)

        for chunk_idx, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            if len(current_tokens) + len(tokens) <= sequence_length:
                current_tokens.extend(tokens)
                current_segment_ids.extend([sequence_number] * len(tokens))
                sequence_number += 1
            else:
                # Save the current stacked sequences
                if len(current_tokens) > 0:
                    save_record(output_dir, record_count, current_tokens, current_segment_ids, sequence_length)
                    record_count += 1
                    sequence_number = 1

                # Start a new sequence
                current_tokens = tokens
                current_segment_ids = [sequence_number] * len(tokens)

            # Check if we have a full sequence to save so we can save immediately
            if len(current_tokens) == sequence_length:
                save_record(output_dir, record_count, current_tokens, current_segment_ids, sequence_length)
                record_count += 1
                current_tokens = []
                current_segment_ids = []
                sequence_number = 1

    # Save any remaining tokens
    if len(current_tokens) > 0:
        save_record(output_dir, record_count, current_tokens, current_segment_ids, sequence_length)


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


def save_record(output_dir, record_count, tokens, segment_ids, sequence_length):
    # Pad if necessary
    if len(tokens) < sequence_length:
        padding_length = sequence_length - len(tokens)
        tokens = np.concatenate([tokens, np.full(padding_length, stoi["P"])])
        segment_ids = segment_ids + [0] * padding_length

    output_file = os.path.join(output_dir, f"record_{record_count}.tfrecord")
    with tf.io.TFRecordWriter(output_file) as writer:
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
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    for batch in dataset:
        yield {
            "x": batch["x"].numpy().astype(np.int32),
            "segment_ids": batch["segment_ids"].numpy().astype(np.int32),
        }
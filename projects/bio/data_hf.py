import os
import re

import numpy as np
import tensorflow as tf
from tqdm import tqdm

VOCAB = ["P", "S", "E", "U", "A", "C", "G", "T"]  #  padding, start, end, unknown, A, C, G, T
VOCAB_SIZE = len(VOCAB)
stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}
encode = lambda x: [stoi.get(ch, 0) for ch in x]
decode = lambda x: "".join([itos.get(i, "U") for i in x])


def preprocess_dna_sequence(x, sequence_length):
    # remove all non ACGT characters and convert to uppercase
    x = re.sub(r"[^ACGT]", "", x.upper())
    start_end_adjusted_seqlen = sequence_length - 2
    # split into chunks of SEQ_LEN, append start/end token.
    return ['S' + x[i : i + start_end_adjusted_seqlen] + 'E' for i in range(0, len(x), start_end_adjusted_seqlen)]


def process_and_save_tfrecords(dataset, output_dir, sequence_length=8192):
    os.makedirs(output_dir, exist_ok=True)
    current_tokens = []
    current_segment_ids = []
    record_count = 0
    sequence_number = 1

    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        sequence = example["text"]
        chunks = preprocess_dna_sequence(sequence, sequence_length)

        for chunk_idx, chunk in enumerate(chunks):
            tokens = encode(chunk)

            if len(current_tokens) + len(tokens) <= sequence_length:
                current_tokens.extend(tokens)
                current_segment_ids.extend([sequence_number] * len(tokens))
                sequence_number += 1
            else:
                # Save the current stacked sequences
                if len(current_tokens) > 0:
                    save_record(output_dir, record_count, current_tokens, current_segment_ids, sequence_length)
                    record_count += 1

                # Start a new sequence
                current_tokens = tokens
                current_segment_ids = [sequence_number] * len(tokens)
                sequence_number += 1

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

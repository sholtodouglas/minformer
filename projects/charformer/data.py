"""Tokenizes text and creates dataloader for datasets too big to fit in memory."""

import numpy as np
from typing import Dict, List, Tuple
import tensorflow as tf
import os
import string
import tqdm
from typing import Any

class CharDataset:
    """
    Emits batches of characters
    """
    @staticmethod
    def get_default_config():
        class Config:
            sequence_length: int = 4096
        return Config()

    def __init__(self, config):
        self.config = config
        # All ASCII chars + some extra ones which appear in this dataset.
        self.chars = list(string.printable) + ['"', '"', ''', '–', '—', ''', 'é', '…', '\xa0', 'ñ', 'à', '´']
        self.stoi: Dict[str, int] = {ch: i+1 for i, ch in enumerate(self.chars)}  # Start from 1
        self.stoi['<unk>'] = 0  # Add unknown token with id 0
        self.itos: Dict[int, str] = {i+1: ch for i, ch in enumerate(self.chars)}  # Start from 1
        self.itos[0] = '<unk>'  # Add unknown token with id 0

    @property
    def vocab_size(self) -> int:
        return len(self.chars)
    
    def tokenize(self, text):
        return np.array([self.stoi.get(c, 0) for c in text], dtype=np.int32)

    @property
    def sequence_length(self) -> int:
        return self.config.sequence_length
    
    @property
    def feature_description(self) -> Any:
        # Define the features to parse from the TFRecord
        return {
            'x': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'y': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'segment_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
    

    def create_packed_records(self, input_file_path: str, output_dir: str, custom_delimiter: str):
        """Takes a (potentially large) file, and saves it into tf records."""

        files_saved_so_far = 0
        token_span = np.zeros((self.sequence_length+1), dtype=np.int32)
        segment_ids = np.zeros((self.sequence_length+1), dtype=np.int32)
        current_text = ""
        tokens_so_far = 0
        segments_so_far = 1

 
        os.makedirs(output_dir, exist_ok=True)

        def save_tfrecord(writer, token_span, segment_ids):
            x = token_span[:-1]
            y = token_span[1:]
            example = tf.train.Example(features=tf.train.Features(feature={
                'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
                'segment_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids))
            }))
            writer.write(example.SerializeToString())

        with open(input_file_path, 'r') as file:
            for i, line in tqdm.tqdm(enumerate(file)):

                if custom_delimiter in line:
                    # If we see the end token.
                    parts = line.split(custom_delimiter)
                    # Add it to the current text.
                    current_text += parts[0] + custom_delimiter
                    # Tokenize it.
                    tokens = self.tokenize(current_text)
                    # Now that we're done with this text, reset it.
                    current_text = ""
                    # Currently this just discards seqs too long, meh.
                    if tokens_so_far + len(tokens) < self.sequence_length:
                        # If it fits into the current sequence, slice it in.
                        
                        token_span[tokens_so_far: tokens_so_far + len(tokens)] = tokens
                        segment_ids[tokens_so_far: tokens_so_far + len(tokens)] = np.ones((len(tokens)), dtype=np.int32) * segments_so_far
                        tokens_so_far += len(tokens)
                        segments_so_far += 1
                    else:
                        # Otherwise save the running record.
                        # In this case, we could never have packed it.
                        if len(tokens) > self.sequence_length:
                            token_span = tokens[:self.sequence_length+1]
                            segment_ids = np.ones_like(token_span)
                        new_output_file = output_dir + f'record_{files_saved_so_far}.tfrecord'
                        # print("".join([self.itos[int(t)] for t in token_span[:]])+"\n\n ------- \n\n")
                        # print(segment_ids[:2048])
                        with tf.io.TFRecordWriter(new_output_file) as writer:
                            save_tfrecord(writer, token_span, segment_ids)
                        # And reset everything.
                        token_span = np.zeros((self.sequence_length+1), dtype=np.int32)
                        segment_ids = np.zeros((self.sequence_length+1), dtype=np.int32)
                        tokens_so_far = 0
                        segments_so_far = 1
                        files_saved_so_far += 1
                else:
                    current_text += line


    def load_and_retokenize_tfrecord(self, file_path: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Loads a TFRecord file and retokenizes its content according to the current CharDataset instance.

        Example filepath: 'projects/charformer/data/tfrecords/record_121.tfrecord'
        
        Args:
        file_path (str): Path to the TFRecord file.
        
        Returns:
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: A list of tuples containing (x, y, segment_ids) for each example.
        """
        retokenized_data = []
    
        
        def _parse_function(example_proto):
            return tf.io.parse_single_example(example_proto, self.feature_description)
        
        dataset = tf.data.TFRecordDataset(file_path)
        parsed_dataset = dataset.map(_parse_function)
        
        for parsed_record in parsed_dataset:
            x = parsed_record['x'].numpy()
            y = parsed_record['y'].numpy()
            segment_ids = parsed_record['segment_ids'].numpy()
            for s in np.unique(segment_ids):
            # Convert token IDs back to characters
                if s != 0:
                    min_max = np.where(segment_ids == s)[0]
                    min_idx, max_idx = min_max[0], min_max[-1]
                    original_text = ''.join([self.itos[token] for token in x[min_idx:max_idx]])
                    retokenized_data.append(original_text)
        
        return retokenized_data
    
    def create_iterator(self, file_pattern: str, batch_size: int, shuffle: bool = False):
        """Creates a python iterator to load batches."""
        def _parse_function(example_proto):
            parsed_features =  tf.io.parse_single_example(example_proto, self.feature_description)
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
                'x': batch['x'].numpy().astype(np.int32),
                'y': batch['y'].numpy().astype(np.int32),
                'segment_ids': batch['segment_ids'].numpy().astype(np.int32)
            }



                




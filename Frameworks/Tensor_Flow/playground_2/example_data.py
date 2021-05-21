import pandas as pd
import tensorflow as tf

import data_generator

COLUMN_NAMES = ['x', 'y','Color']
NUM_TRAIN_DATA = 200
NUM_TEST_DATA = 30

def load_data(y_name='Color'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_data = data_generator.generate(NUM_TRAIN_DATA)
    print("TRAIN DATA:")
    print(train_data)
    train = pd.DataFrame(train_data, columns = COLUMN_NAMES)
    train_x, train_y = train, train.pop(y_name)

    test_data = data_generator.generate(NUM_TEST_DATA)
    print("TEST DATA:")
    print(test_data)
    test =  pd.DataFrame(test_data, columns = COLUMN_NAMES)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Color')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

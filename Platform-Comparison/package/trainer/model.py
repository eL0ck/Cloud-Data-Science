from __future__ import print_function
import os
import pandas as pd
import tensorflow as tf
# import tensorflow.estimator.inputs as tfi  # Import error TF 1.8.0 ?!
tfi = tf.estimator.inputs


# ------------------- DEFINE THE TF MODEL ------------------------------------- #

BATCH_SIZE = 100
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn(*a, **k):
    feature_placeholders = {feature: tf.placeholder(tf.float32, shape=[None])
        for feature in CSV_COLUMN_NAMES[:-1]  # Not 'Species'
    }
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def estimator_fn(run_config=None, params=None, output_dir=None):
    my_feature_columns = [tf.feature_column.numeric_column(key)
        for key in CSV_COLUMN_NAMES[:-1]  # Not 'Species'
    ]

    return tf.estimator.DNNClassifier(
        model_dir=output_dir,
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3,  # 3 species of iris to group by
        config=run_config,
    )


def _input_fn(mode, data_path, data_filename):
    """
        Args:
            mode ('eval'/'train'):
            data_path (str): S3 or local path to test and training 
                files.  Wont accept GCS files.
        Returns: TensorFlow input function
    """

    if mode == 'train':
        shuffle = True
        num_epochs = 500
    elif mode == 'eval':
        shuffle = False
        num_epochs = 1
    else:
        raise Exception('invalid mode: {}'.format(mode))

    data_file = os.path.join(data_path, data_filename)
    print("Expecting data from: ", data_file)

    df = pd.read_csv(data_file, names=CSV_COLUMN_NAMES, header=0)
    df_x, df_y = df, df.pop('Species')

    return tfi.pandas_input_fn(
        df_x, df_y, BATCH_SIZE,
        shuffle=shuffle, num_epochs=num_epochs,
    )


# ----------------- Sagemaker input Functions ------------------------------- #
# N.B The functions are actually called here!

def train_input_fn(training_dir, hyperparams):
    """Returns input function that would feed the model during training"""
    return _input_fn('train', training_dir, 'iris_training.csv')()  


def eval_input_fn(training_dir, hyperparams):
    """Returns input function that would feed the model during evaluation"""
    return _input_fn('eval', training_dir, 'iris_test.csv')()


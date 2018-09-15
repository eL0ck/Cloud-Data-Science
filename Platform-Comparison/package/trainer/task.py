from __future__ import print_function
import os
import json
from google.cloud import storage
import argparse
import tensorflow as tf

from model import _input_fn, estimator_fn, serving_input_fn



tf.logging.set_verbosity(tf.logging.INFO)


def _get_data_from_gs(config):
    for gs_file_name in (config['train_file'], config['test_file']):
        print('Downloading: gs://{}/{}'.format(config['bucket'], gs_file_name))
        blob = storage.Client(config['project']).get_bucket(config['bucket']).blob(gs_file_name)
        
        # Make the full path
        path = os.path.join('/tmp', os.path.dirname(gs_file_name))
        if not os.path.exists(path):
            os.makedirs(path)
            
        blob.download_to_filename(os.path.join('/tmp', gs_file_name))


def train_and_eval(config):
    """ Run model from here"""

    est = estimator_fn(output_dir=config['outdir'])

    print('Defining training spec')
    train_spec = tf.estimator.TrainSpec(
        input_fn=_input_fn('train', data_path='/tmp', data_filename=config['train_file']),
        max_steps=config['train_steps'],
    )

    print('Defining eval spec')
    eval_spec = tf.estimator.EvalSpec(
        input_fn=_input_fn('eval', data_path='/tmp', data_filename=config['test_file']),
        steps=None,
        start_delay_secs=0,
        throttle_secs=1,
        exporters=tf.estimator.LatestExporter('exporter', serving_input_fn),
    )

    print('Starting training ...')
    tf.estimator.train_and_evaluate(est, train_spec, eval_spec)


def main(config):    
    print('Running Tensorflow version:', tf.__version__)
    print('Received OUTDIR:', config['outdir'])
    _get_data_from_gs(config)    
    train_and_eval(config)   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--outdir',
        help='GCS or local path to model artefacts',
        required=True,
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk',
    )
    parser.add_argument(
        '--train_steps',
        help='Number of to perform training over',
        required=True,
        type=int,
        default=10,
    )
    parser.add_argument(
        '--project',
        help='Project with the data bucket.',
        required=True,
    )
    parser.add_argument(
        '--bucket',
        help='Test and training data bucket name.',
        required=True,
    )
    parser.add_argument(
        '--test_file',
        help='Test file name within the bucket.',
        required=True,
    )
    parser.add_argument(
        '--train_file',
        help='Training file name within the bucket.',
        required=True,
    )    
    
    kwarguments = parser.parse_args().__dict__

    # Unused args provided by service
    kwarguments.pop('job_dir', None)
    kwarguments.pop('job-dir', None)

    outdir = kwarguments['outdir']
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    outdir = os.path.join(
        outdir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trail', '')
    )
    
    main(kwarguments)
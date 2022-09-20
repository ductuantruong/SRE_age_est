import os
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)

class ModelConfig(object):
    
    dir = config['dataDir']['dir']
    
    # path to the unzipped TIMIT data folder
    data_path = config['dataDir']['data_path'].replace('$dir', dir)

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = config['dataDir']['speaker_csv_path'].replace('$dir', dir)

    batch_size = int(config['model_parameters']['batch_size'])
    
    epochs = int(config['model_parameters']['epochs'])

    model_type = config['model_parameters']['model_type']

    # UncertaintyLoss
    loss = "UncertaintyLoss"
    
    pretrained_upstream_path = config['model_parameters']['pretrained_upstream_path']

    # number of layers in encoder (transformers)
    num_layers = 6

    # feature dimension of upstream model. For example, 
    # For wav2vec2, feature_dim = 768
    feature_dim = int(config['model_parameters']['feature_dim'])

    # No of GPUs for training and no of workers for datalaoders
    gpu = int(config['gpu'])
    n_workers = int(config['n_workers'])

    # model checkpoint to continue from
    model_checkpoint = None
    
    # LR of optimizer
    lr = float(config['model_parameters']['lr'])
    
    narrow_band = config['model_parameters']['narrow_band']

    run_name = 'multi-task' + '_' + model_type

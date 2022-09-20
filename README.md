# Speaker Profiling

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses TIMIT dataset. 

## Installation
### Setting up kaldi environment
```
git clone -b 5.4 https://github.com/kaldi-asr/kaldi.git kaldi
cd kaldi/tools/; make; cd ../src; ./configure; make
```

### Setting up python environment

```bash
pip install -r requirements.txt
```

## Generate wav files
Because wav2vec2 model training can only be sent to the absolute path of audio files, not in the form of pipes.\
So the first step is to regenerate a new wav file using the original wav.scp and segments files.\
You can use the script to prepare all the data, but it may take a long time to generate all new wav file.
```
bash scripts/prepare_wav_file.sh
```

## Fine-tuning wav2vec 2.0
### Download pretrained wav2vec2 model
Instead of training from scratch, we download and use english wav2vec model for weight initialization. This pratice can be apply to all languages.
```
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
```
### Use new data to finetune orignal wav2vec2 model
```
python3 finetune_w2v2/pretrain.py --fairseq_path finetune_w2v2/fairseq/ --audio_path data/train/wav_path --init_model wav2vec_small.pt
```
After the training, you can find an 
Example:
```
outputs/2021-09-02/00-04-52/checkpoints/checkpoint_best.pt
```
Move the finetuned wav2vec 2.0 to the project directory:
```
mv finetune_w2v2/outputs/2021-09-02/00-04-52/checkpoints/checkpoint_best.pt $PWD/sre_50epchs_finetuned_w2v2.pt
```

## Usage

### Prepare the dataset for training and testing
```bash
python SRE/prepare_sre_data.py --path='path to timit data folder'
```

### Update Config and Logger
Update the config.json file to update the upstream model, batch_size, gpus, lr, etc and change the preferred logger in train_.py files. Create a folder 'checkpoints' to save the best models. If you wish to perform narrow band experiment, just set narrow_band as true in config.json file.

### Training
```bash
python train_sre.py --data_path='path to final data folder' --speaker_csv_path='path to this repo/SpeakerProfiling/Dataset/data_info_height_age.csv'
```

Example:
```bash
python train_sre.py --data_path=data/ --speaker_csv_path=Dataset/data_info_age.csv
```

### Testing
```bash
python test_sre.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_sre.py --data_path=data/ --model_checkpoint=checkpoints/epoch=7-step=13647.ckpt
```

### Pretrained Model
We have uploaded a pretrained model of our experiments. You can download the from [Dropbox](https://www.dropbox.com/s/xkijgjhlht5pfd1/epoch%3D7-step%3D13647.ckpt?dl=0).

Download it and put it into the model_checkpoint folder.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen


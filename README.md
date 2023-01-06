# Speaker Profiling

This Repository contains the code for estimating the Age and Height of a speaker with their speech signal. This repository uses [s3prl](https://github.com/s3prl/s3prl) library to load various upstream models like wav2vec2, CPC, TERA etc. This repository uses SRE dataset. 

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

## Upsample wav files
Because most of SSL upstream models are trained on 16kHz audio files. However, SRE08/10 audios are only in 8kHz. Therefore, we need to upsample them by running the following script.
```
python SRE/upsample.sh
```

## Usage

### Prepare the dataset for training and testing
```bash
python SRE/prepare_sre_data.py --path='path to SRE data folder'
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
We have uploaded a pretrained model of our experiments. You can download the from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/ductuan001_e_ntu_edu_sg/EuH_IItaGnFPj4O9nqd5190Bk39DX3CYAc0h3bu5wn3ppQ?e=AatgN2).

Download it and put it into the model_checkpoint folder.

## Testing with Simulated Noisy Speech
### Download the MUSAN dataset
```bash
wget https://www.openslr.org/resources/17/musan.tar.gz
tar -xvzf musan.tar.gz
```

### Prepare the simulated noisy dataset with 5 noisy level: 5dB, 10dB, 15dB and 20dB.
```bash
python SRE/create_noisy_data.py --data_path='path to SRE data folder' --noise_path='path to SRE data folder' --noisy_type='type of noise in musan subfolder'
```

Example:
```bash
python SRE/create_noisy_data.py --data_path=data --noise_path=data/musan --noisy_type=music
```
### Run test on the noisy test set
```bash
python test_sre.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_sre.py --data_path=data/noisy_data_music/ --model_checkpoint=checkpoints/epoch=7-step=13647.ckpt
```

## Testing with DSO test set
### Run test on the noisy test set
```bash
python test_dso.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

Example:
```bash
python test_dso.py --data_path=data/DSO_data/ --test_speaker_csv_path=DSO/data_info_height_age.csv  --model_checkpoint=checkpoints/epoch=7-step=13647.ckpt
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S3prl: The self-supervised speech pre-training and representation learning toolkit. AT Liu, Y Shu-wen


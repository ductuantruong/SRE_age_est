import torch
import torchaudio
from tqdm import tqdm
from multiprocess import Pool
import os
import argparse

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--data_path',
                       metavar='data_path',
                       default='data/NIST_SRE_Corpus/',
                       type=str,
                       help='the path to dataset folder')

args = my_parser.parse_args()

DATA_TYPE = 'test'
SAVED_DIR = os.path.join(args.data_path, DATA_TYPE + '_16k')

if not os.path.exists(SAVED_DIR):
    os.mkdir(SAVED_DIR)

resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

def upsample_wav(wav_file):
    wav, _ = torchaudio.load(os.path.join(args.data_path, DATA_TYPE, wav_file))
    new_wav = resampleUp(wav)
    torchaudio.save(os.path.join(SAVED_DIR, wav_file), new_wav, 16000)



list_wav_file = os.listdir(os.path.join(args.data_path, DATA_TYPE))

with Pool(processes=6) as pool:
    pool.map(upsample_wav, tqdm(list_wav_file))



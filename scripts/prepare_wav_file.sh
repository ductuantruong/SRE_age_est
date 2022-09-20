#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

data=data-org
tgt=data
train_set="train"
valid_set="valid"
recog_set="test"

echo "## Generate new wav file for sre08/10 dataset"
for x in $valid_set $recog_set; do
   python scripts/generate_new_wav.py data-org/$x/wav.scp data-org/$x/segments $PWD/data/NIST_SRE_Corpus/$x/ > data-org/$x/generate_cmd.sh
   mkdir $tgt/$x -p
   cp $data/$x/{text,utt2spk,utt2age} $tgt/$x
   path="$PWD/data/NIST_SRE_Corpus/"$x
   cat $data/$x/utt2spk | awk -v p="$path" '{print $1 " "p"/"$1".wav"}' > $tgt/$x/wav.scp
   cat $tgt/$x/wav.scp | cut -d ' ' -f2 > $tgt/$x/wav_path
   utils/fix_data_dir.sh $tgt/$x
   utils/validate_data_dir.sh --no-feats $tgt/$x
done

echo "## Generate new wav file for sre08/10 dataset Done"

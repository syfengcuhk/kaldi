#!/bin/bash
current_path=$(pwd)
kaldi_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related

#data_path=data/train_unlab_600/
data_path=data/train_unlab_6k/
mkdir -p $data_path

#data_path=data/dev-other/
#data_path=data/dev-clean/
#data_path=data/test-other/
#data_path=data/test-clean/
#data_path=data/unseg_train_unlab_600
#data_path=data/unseg_train_unlab_6k_cut
  cd $data_path
  spk_pos=12 # 11 for train_unlab_600, 12 for dev-{clean,other} and test-{clean,other} and unseg_train
  # genenrate flist_w_path.txt and flist.txt
#  cd /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/data_unlab/unlab-6k_cut
#  echo "current dir at $pwd"
#  find  "$(pwd)" -name *.flac > $kaldi_path/$data_path/flist_w_path.txt
#  find -type f -name *.flac | sed 's:.*/::' > $kaldi_path/$data_path/flist.txt
  #cd $current_path 
  cut -d'/' -f${spk_pos} flist_w_path.txt > spklist.txt
  cat ../train_unlab_600/spklist.txt spklist.txt > spklist_s_m.txt
  sed 's/\.flac//g' flist.txt > uttlist.txt
  cat ../train_unlab_600/uttlist.txt uttlist.txt > uttlist_s_m.txt
  cat ../train_unlab_600/flist_w_path.txt flist_w_path.txt > flist_w_path_s_m.txt
  paste -d' ' uttlist_s_m.txt spklist_s_m.txt | paste -d'-' spklist_s_m.txt -   > utt2spk
  #$cd $current_path
  cd ../../
  ./utils/utt2spk_to_spk2utt.pl $data_path/utt2spk > $data_path/spk2utt
  cd $data_path
  #sort spk2utt > spk2utt.bak
  #mv spk2utt.bak spk2utt
  sort utt2spk > utt2spk.bak
  #mv utt2spk > .utt2spk
  mv utt2spk.bak  utt2spk
  paste -d' ' uttlist_s_m.txt flist_w_path_s_m.txt | paste -d'-' spklist_s_m.txt - > wav.scp 

# then add flac -c -d -s xxxx and add | in the end
#for set in train_unlab_600 unseg_train_unlab_600 dev-other dev-clean test-other test-clean; do
cd $current_path
for set in train_unlab_6k; do
#for set in dev-other dev-clean test-other test-clean; do
  sed -e 's/\.flac/\.flac |/g' -e 's/\/tudelft\.net/ flac -c -d -s \/tudelft\.net/g' data/$set/wav.scp > data/$set/wav.scp.new || exit 1
   #sed  -e 's/\/tudelft\.net/ flac -c -d -s \/tudelft\.net/g' data/$set/wav.scp > data/$set/wav.scp.new
   mv data/$set/wav.scp.new data/$set/wav.scp

done

echo "succeeded..."
cd $current_path

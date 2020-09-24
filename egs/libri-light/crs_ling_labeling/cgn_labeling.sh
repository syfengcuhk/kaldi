#!/bin/bash
align_fmllr_stage=0
cmd=run.pl
acwt=0.1
make_fbank_nj=1
beam=15
lattice_beam=8
decode_fmllr_stage=0
post_decode_acwt=10.0
train_name=train_unlab_600 # or train_unlab_600_subset3600utt
suffix_decode_dir=
nj=1
num_threads_decode=1
decode_nj=1
stage=0
stop_stage=1
model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/chain/bn_layer/tdnn1a_sp_bi_epoch3_inputmfcc_hires_only
gmm_model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/train/tri4
graph_dir=$model/graph
eval_set_appoint=dev-clean
. ./path.sh
. ./utils/parse_options.sh
if [ $stage -le -1 ]  && [ $stop_stage -gt -1 ];then
    echo "$0: extracting ivectors"
    ivector_extractor=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/nnet3/extractor
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_cgn
    input_data=$data_root_path/$train_name
    output_data=$ivector_extractor/../ivectors_libri_light_hires/$train_name
    steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
        $input_data $ivector_extractor $output_data || exit 1;
fi
if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
    echo "$0: decoding libri-light $train_name to get lattices"
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_cgn
    input_data=$data_root_path/$train_name
    decoding_dir=$model/decoding_for_libri_light_${train_name}${suffix_decode_dir}_acwt${acwt} #lets put its root dir where final.nnet locates 
    input_ivec_dir=$model/../../../nnet3/ivectors_libri_light_hires/$train_name
    nspk=$(wc -l <$input_data/spk2utt)
    [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
    steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk \
       --cmd run.pl  --online-ivector-dir $input_ivec_dir \
       --skip-scoring true --acwt ${acwt} --post-decode-acwt ${post_decode_acwt} \
        $graph_dir $input_data $decoding_dir || exit 1; 
fi

#if [ $stage -le 1 ] && [ $stop_stage -gt 1 ];then
#   echo "$0: decoding libri-light $train_name with $gmm_model"
#   input_data=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data/$train_name
#   decoding_dir=$gmm_model/decode_libri_light_${train_name}
#   graph_dir=$gmm_model/graph_tgpr
#   nspk=$(wc -l <$input_data/spk2utt)
#   [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
#   steps/decode_fmllr.sh --stage $decode_fmllr_stage --skip-scoring true --num-threads $num_threads_decode  --nj $nspk $graph_dir $input_data $decoding_dir || exit 1;
#fi

#if [ $stage -le 2 ] && [ $stop_stage -gt 2 ];then
#  echo "$0: debug decode libri-light dev-clean with $model"
#  set_name=dev-clean
#  data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_cgn
#  input_data=$data_root_path/$set_name
#  #decoding_dir=$model/decode_libri_light_${set_name}_indeed_dev_s
#  decoding_dir=$model/decode_libri_light_${set_name}
#  input_ivec_dir=$model/../../../nnet3/ivectors_libri_light_hires/$set_name
#  ##### do debug with cgn dev_s data
#  #input_data=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/data/dev_s_hires
#  #input_ivec_dir=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/nnet3/ivectors_dev_s_hires
#  #########
#  nspk=$(wc -l <$input_data/spk2utt)
#  [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
#  steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk \
#     --beam $beam --lattice-beam $lattice_beam --cmd run.pl  --online-ivector-dir $input_ivec_dir \
#     --skip-scoring true --acwt 1.0 --post-decode-acwt ${post_decode_acwt} \
#      $graph_dir $input_data $decoding_dir || exit 1;
#
#fi

#if [ $stage -le 3 ] && [ $stop_stage -gt 3 ];then
#    echo "$0: make fbank features to prepare for cgn: exp/train/tri4_dnn_mpe decoding"
#  for set in  train_unlab_600_subset3600utt train_unlab_600_subset7200utt train_unlab_600_subset14400utt train_unlab_600 ; do
#    mkdir -p data_fbank_conf_cgn/$set || exit 1;
#    echo "$0: stage 3, proc. $set "
#    utils/copy_data_dir.sh data/$set data_fbank_conf_cgn/$set
#    nspk=$(wc -l < data_fbank_conf_cgn/$set/spk2utt)
#    [ "$nspk" -gt "$make_fbank_nj" ] && nspk=$make_fbank_nj
#    steps/make_fbank.sh --nj $nspk --fbank-config conf/fbank_from_cgn.conf data_fbank_conf_cgn/$set || exit 1;
#    steps/compute_cmvn_stats.sh data_fbank_conf_cgn/$set || exit 1
#    #utils/validate_data_dir/sh data_fbank_conf_cgn/$set
#  done
#fi
#
#if [ $stage -le 4 ] && [ $stop_stage -gt 4 ];then
#  echo "$0: decode ligri-light subset3600utt fbank40 by cgn:exp/train/tri4_dnn_mpe"
#   mpe_iter=2
#   nnet1_model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/train/tri4_dnn_mpe
#   gmmdir_for_graph=$nnet1_model/../../train_s/tri3
#   input_data=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_fbank_conf_cgn/$train_name
#   decoding_dir=$nnet1_model/decode_libri_light_${train_name}_it${mpe_iter}_acwt${acwt}
#   graph_dir=$gmmdir_for_graph/graph_tgpr
#   nspk=$(wc -l <$input_data/spk2utt)
#   [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
#   steps/nnet/decode.sh --skip-scoring true --num-threads $num_threads_decode --nj $nspk --nnet $nnet1_model/${mpe_iter}.nnet --config $nnet1_model/../../../conf/decode_dnn.config --acwt $acwt $graph_dir $input_data $decoding_dir || exit 1; 
#fi

if [ $stage -le 5 ] && [ $stop_stage -gt 5 ];then
  echo "$0: convert to text by lattice-best-path"
  input_dir=$model/decoding_for_libri_light_${train_name}${suffix_decode_dir}_acwt${acwt}
  num_jobs=$(cat $input_dir/num_jobs) || exit 1;
  $cmd JOB=1:$num_jobs $input_dir/log/lattice_best_path.JOB.log \
          lattice-best-path --lm-scale=0.001 "ark:gunzip -c $input_dir/lat.JOB.gz|" "ark,t:|utils/int2sym.pl -f 2- $input_dir/../graph/words.txt >   $input_dir/text.JOB" || exit 1;
  cat $input_dir/text.* > $input_dir/text || exit 1; 
fi
if [ $stage -le 6 ] && [ $stop_stage -gt 6 ];then
  echo "$0: construct data files with transcriptions"
  text_dir=$model/decoding_for_libri_light_${train_name}${suffix_decode_dir}_acwt${acwt}/
  utils/copy_data_dir.sh data/${train_name} data_plus_cgn_transcripts/acwt${acwt}/${train_name} || exit 1;
  cp $text_dir/text data_plus_cgn_transcripts/acwt${acwt}/${train_name}
  utils/fix_data_dir.sh data_plus_cgn_transcripts/acwt${acwt}/${train_name}
fi

if [ $stage -le 7 ] && [ $stop_stage -gt 7 ];then
  echo "$0: get alignment for tree building"
  data_dir=data_plus_cgn_transcripts/acwt${acwt}/${train_name}
  ali_dir=exp/chain_cgn_lat_label/$train_name/lat_gen_acwt${acwt}/${train_name}_ali
  steps/align_fmllr.sh --nj $nj --cmd run.pl --stage $align_fmllr_stage  \
    $data_dir $gmm_model/../../../data/lang_s $gmm_model $ali_dir
fi

if [ $stage -le 8 ] && [ $stop_stage -gt 8 ];then
  echo "$0: decoding libri-light {dev,test}-{clean,other} to get lattices, followed by getting best path"
#  echo "$0: get alignment "
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_cgn
    for set in dev-clean dev-other test-clean test-other; do
      input_data=$data_root_path/$set
      decoding_dir=$model/decoding_for_libri_light_${set}${suffix_decode_dir}_acwt${acwt} #lets put its root dir where final.nnet locates 
      input_ivec_dir=$model/../../../nnet3/ivectors_libri_light_hires/$set
      nspk=$(wc -l <$input_data/spk2utt)
      [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
      steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk \
         --cmd run.pl  --online-ivector-dir $input_ivec_dir \
         --skip-scoring true --acwt ${acwt} --post-decode-acwt ${post_decode_acwt} \
          $graph_dir $input_data $decoding_dir || exit 1;
    done
fi

if [ $stage -le 9 ] && [ $stop_stage -gt 9 ];then
  echo "$0: eval data: convert to text by lattice-best-path"
  for set in dev-clean dev-other test-clean test-other; do
    input_dir=$model/decoding_for_libri_light_${set}${suffix_decode_dir}_acwt${acwt}
    num_jobs=$(cat $input_dir/num_jobs) || exit 1;
    $cmd JOB=1:$num_jobs $input_dir/log/lattice_best_path.JOB.log \
            lattice-best-path --lm-scale=0.001 "ark:gunzip -c $input_dir/lat.JOB.gz|" "ark,t:|utils/int2sym.pl -f 2- $input_dir/../graph/words.txt >   $input_dir/text.JOB" || exit 1;
    cat $input_dir/text.* > $input_dir/text || exit 1;
  done
fi

if [ $stage -le 10 ] && [ $stop_stage -gt 10 ];then
  echo "$0: eval data construct data files with transcriptions"
  for set in dev-clean dev-other test-clean test-other; do
    text_dir=$model/decoding_for_libri_light_${set}${suffix_decode_dir}_acwt${acwt}/
    utils/copy_data_dir.sh data/${set} data_plus_cgn_transcripts/acwt${acwt}/${set} || exit 1;
    cp $text_dir/text data_plus_cgn_transcripts/acwt${acwt}/${set}
    utils/fix_data_dir.sh data_plus_cgn_transcripts/acwt${acwt}/${set}
  done
fi 
if [ $stage -le 11 ] && [ $stop_stage -gt 11 ];then
  echo "$0: eval data get alignment"

#  for set in dev-clean dev-other test-clean test-other; do
    set=$eval_set_appoint
    data_dir=data_plus_cgn_transcripts/acwt${acwt}/${set}
    ali_dir=exp/chain_cgn_lat_label/$set/lat_gen_acwt${acwt}/${set}_ali_nj4
    steps/align_fmllr.sh --nj 4 --cmd run.pl --stage $align_fmllr_stage  \
      $data_dir $gmm_model/../../../data/lang_s $gmm_model $ali_dir
#  done
fi

if [ $stage -le 12 ] && [ $stop_stage -gt 12 ];then
  echo "$0: convert to ctm files"
#    for set in dev-clean dev-other test-clean test-other; do
      set=$eval_set_appoint
      dir=exp/chain_cgn_lat_label/$set/lat_gen_acwt${acwt}/${set}_ali_nj4
      num_jobs=$(cat $dir/num_jobs) || exit 1;
      $cmd JOB=1:$num_jobs $dir/log/get_ctm.JOB.log \
       ali-to-phones --ctm-output $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" "|utils/int2sym.pl -f 5- $dir/phones.txt > $dir/ctm.JOB.txt " || exit 1;
      cat $dir/ctm.*.txt > $dir/ctm.txt || exit 1;

      $cmd JOB=1:$num_jobs $dir/log/get_ctm_int.JOB.log \
       ali-to-phones --ctm-output $dir/final.mdl "ark:gunzip -c $dir/ali.JOB.gz|" $dir/ctm_int.JOB.txt || exit 1;
      cat $dir/ctm_int.*.txt > $dir/ctm_int.txt || exit 1
#    done
fi
echo "succeeded"

#!/bin/bash
nj=1
stage=0
stop_stage=1
model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/aidatatang_200zh/s5/exp/chain/bnf_layer/tdnn_1a_input_no_ivec_sp
abx_eval_set_appoint=dev-clean
dur_appoint=1s
lang_appoint=english
. ./path.sh
. ./utils/parse_options.sh
if [ $stage -le -1 ]  && [ $stop_stage -gt -1 ];then
    echo "$0: extracting ivectors"
    ivector_extractor=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/aidatatang_200zh/s5/exp/nnet3/extractor
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_aidatatang # this is no pitch version of features
    for set in  dev-other dev-clean test-other test-clean; do
        input_data=$data_root_path/$set
        output_data=$ivector_extractor/../ivectors_libri_light_hires/$set
        steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
	    $input_data $ivector_extractor $output_data || exit 1;
    done 
fi
#model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/chain/bn_layer/tdnn1a_sp_bi_epoch3_inputmfcc_hires_only
bnf_name=tdnn6.affine
if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_pitch_conf_aidatatang # this time we need mfcc_hires_pitch features to chain-dnn inputs
    #bnf_name=tdnn6
    for set in  dev-other dev-clean test-other test-clean; do
        input_data=$data_root_path/$set
        output_data=$model/bnf_for_libri_light/$set/
        input_ivec_dir=$model/../../../nnet3/ivectors_libri_light_hires/$set
        steps/nnet3/make_bottleneck_features.sh \
           --cmd run.pl  --nj $nj --ivector-dir $input_ivec_dir \
	   $bnf_name $input_data $output_data $model || exit 1; 
    done
fi
#
#if [ $stage -le 1 ] && [ $stop_stage -gt 1 ];then
#   for set in train_unlab_600 unseg_train_unlab_600 ; do
#       steps/make_mfcc.sh --nj $nj data/$set || exit 1;
#       steps/compute_cmvn_stats.sh data/$set || exit 1;
#   done
#fi
#
#if [ $stage -le 2 ] && [ $stop_stage -gt 2 ];then
#   for set in train_unlab_6k  ; do
#       steps/make_mfcc.sh --nj $nj data/$set || exit 1;
#       steps/compute_cmvn_stats.sh data/$set || exit 1;
#   done
#fi
#
if [ $stage -le 3 ] && [ $stop_stage -gt 3 ];then
   for set in dev test; do
      for cond in clean other; do
	 # (
 	  echo "starting ${set}-${cond}"
          data_file=${set}-${cond}
          input_data=$model/bnf_for_libri_light/$data_file
	  output_path=$model/bnf_for_libri_light_by_utt/$data_file
          foo_path=$output_path/../
          mkdir -p $output_path
	  cp data/$data_file/foo_template.scp $foo_path/foo_${set}-${cond}.scp
   	  sed -i "s| PathToRep| $output_path\/|g"  $foo_path/foo_${set}-${cond}.scp || exit 1
	  #apply-cmvn --utt2spk=ark:data/$data_file/utt2spk scp:data/$data_file/cmvn.scp scp:data/$data_file/feats.scp scp,t:$foo_path/foo_${set}-${cond}.scp || exit 1;
	  copy-feats scp:$input_data/feats.scp scp,t:$foo_path/foo_${set}-${cond}.scp || exit 1;
          echo "${set}-${cond}: starting removing [ and ], will construct .npy.txt files and remove .n ones"
          if [ ! -z $output_path ] ; then
                for file in $output_path/*.n ; do
                    sed -e "/\[/d" -e "s/\]//g" -e "s/^ //g" -e "s/ $//g"  $file  > ${file}py.txt;
                done
                # mkdir -p $output_path/fea_files;
                # rm -f $output_path/*.fea || exit 1;
                find $output_path/ -name "*.n" -delete
		find $output_path/ -name "*.npy" -delete
          fi
 	  echo "${set}-${cond} .npy.txt finished"
	  echo "constructing .npy format "
	  python baseline/convert_txt_npy.py $output_path 

	#  ) &
      done
   done

fi

if   [ $stage -le 5 ] && [ $stop_stage -gt 5 ]; then
   echo "$0: stage 5, evaluated by ABX"
   #conda activate libri-light
   echo "enter (libri-light)"
   source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light  # enter env: (libri-light)
   module load cuda/10.0
   PYTHONPATH=    # otherwise torchaudio cannot be imported
   echo "reset PYTHONPATH, value=$PYTHONPATH"
   libri_light_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/
   #for eval_set in dev-clean dev-other test-clean test-other; do
   #for eval_set in dev-clean ; do
   eval_set=$abx_eval_set_appoint
   source_dir=$model/bnf_for_libri_light_by_utt/${eval_set}_npy/
   target_dir=$model/bnf_for_libri_light_by_utt/eval/$eval_set
   mkdir -p $target_dir || exit 1;
   #reason use python3 instead of python is that in ~/.bash_profile I add alias python=@/anaconda3/bin/python, not in @anaconda3/envs/libri-light/bin/python
   python3 ${libri_light_root}/eval/eval_ABX.py $source_dir/ ${libri_light_root}/eval/ABX_src/ABX_data/${eval_set}.item --file_extension .npy --out $target_dir/ --cuda

   #done
   #conda deactivate
   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
   module load cuda/8.0 # reset
   #. ./path.sh
   PYTHONPAT=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH # reset
fi

echo "succeeded"

if [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
  # for extracting bnf features for zrsc2017 test data
  data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/kaldi_stuff/data_hires_pitch_conf_aidatatang
  for lang in english french mandarin; do
    for dur in 1s 10s 120s; do
      input_data=$data_root_path/test/$lang/$dur
      input_ivec_dir=$model/../../../nnet3/ivectors_zr17_hires/test/$lang/$dur
      output_data=$model/bnf_for_zr17/test/$lang/$dur
      steps/nnet3/make_bottleneck_features.sh \
           --cmd run.pl  --nj $nj --ivector-dir $input_ivec_dir \
           $bnf_name $input_data $output_data $model || exit 1;
    done
  done

fi

if [ $stage -le 7 ] && [ $stop_stage -gt 7 ];then
   for lang in english french mandarin; do
      for dur in  $dur_appoint; do
         # (
          echo "starting ${lang}-${dur}"
          input_data=$model/bnf_for_zr17/test/$lang/$dur
          output_path=$model/bnf_for_zr17_by_utt/$lang/$dur
          foo_path=$output_path/../
          mkdir -p $output_path
          zr17_data=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/kaldi_stuff/data/
          cp $zr17_data/test/${lang}_${dur}_template_foo.scp $foo_path/foo_${lang}-${dur}.scp
          sed -i "s| PathToRep| $output_path\/|g"  $foo_path/foo_${lang}-${dur}.scp || exit 1
          #apply-cmvn --utt2spk=ark:data/$data_file/utt2spk scp:data/$data_file/cmvn.scp scp:data/$data_file/feats.scp scp,t:$foo_path/foo_${lang}-${dur}.scp || exit 1;
          copy-feats scp:$input_data/feats.scp scp,t:$foo_path/foo_${lang}-${dur}.scp || exit 1;
          time_stamp_file=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/Autoregressive-Predictive-Coding/zr17_data/time_stamp_${dur}.txt
          if [ ! -z $output_path ] ; then
                for file in $output_path/*.n ; do
                    sed -e "/\[/d" -e "s/\]//g" -e "s/^ //g" -e "s/ $//g"  $file | paste -d' ' $time_stamp_file - > ${file:0:-1}feat  # ${file}py.txt;
#                   paste -d' ' $time_stamp_file ${}        
                done
                # mkdir -p $output_path/fea_files;
                # rm -f $output_path/*.fea || exit 1;
                find $output_path/ -name "*.n" -delete
#                find $output_path/ -name "*.npy" -delete
          fi
          echo "${lang}-${dur}  finished"
#          echo "constructing .npy format "
#          python baseline/convert_txt_npy.py $output_path

        #  ) &
      done
   done

fi
if  [ $stage -le 8 ] && [ $stop_stage -gt 8 ];then
   echo "ABX evaluation"
   echo "$0: $(hostname)"
   echo "enter (zerospeech)"
   source /scratch/siyuanfeng/software/anaconda3/bin/activate zerospeech  # enter env: (libri-light)
   zr17_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017
    lang=$lang_appoint
    dur=$dur_appoint
    eval_set=${lang}_${dur}
    data_file=${lang}_${dur}
    source_dir=$model/bnf_for_zr17_by_utt/$lang/$dur
    target_dir=$source_dir/../../eval/$data_file
    echo "$0: target dir: $target_dir"
    echo "$0: source dir: $source_dir"
    mkdir -p $target_dir || exit 1;
    ls $source_dir/*.feat | wc -l
    echo "nj=$nj"
    python ${zr17_root}/track1/eval/eval_track1.py -j $nj -n 1 $lang ${dur:0:-1} $zr17_root/data $source_dir/ $target_dir/  # duration is like 120s, we need 120 instead

   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
fi

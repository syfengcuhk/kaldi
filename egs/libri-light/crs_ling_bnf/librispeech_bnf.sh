#!/bin/bash
# This script in fact extract phoneme posterior features, instead of BNF.
# the Librispeech TDNN AM is used to propagate acoustic features of dev/test-clean/other sets,
# and convert lattices to posterior-> posterior to phoneme posterior

nj=1
stage=0
stop_stage=1
lm_used=tgsmall
lmscore=0.1
distance_mode_abx=euclidian
model=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/librispeech/ctc/exp/nnet3/tdnn_sp #/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/chain/bn_layer/tdnn1a_sp_bi_epoch3_inputmfcc_hires_only
abx_eval_set_appoint=dev-clean
. ./path.sh
. ./utils/parse_options.sh
if [ $stage -le -1 ]  && [ $stop_stage -gt -1 ];then
    echo "$0: extracting ivectors"
    ivector_extractor=$model/../extractor #/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/exp/nnet3/extractor
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_librispeech
    for set in  dev-other dev-clean test-other test-clean; do
        input_data=$data_root_path/$set
        output_data=$ivector_extractor/../ivectors_libri_light_hires/$set
        steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
	    $input_data $ivector_extractor $output_data || exit 1;
    done 
fi

if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
    data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_librispeech
    #bnf_name=tdnn6
    bnf_name=tdnn5.affine #although it has 650 dimensions
    for set in dev_clean  dev_other test_clean test_other; do
#       lats_dir=$model/decode_${set}_${lm_used}
#       #lmscore=0.1
#       phone_post_dir=$lats_dir/phone_post_lm$lmscore
#       mkdir -p $phone_post_dir
#       lattice-to-post --acoustic-scale=1.0 --lm-scale=$lmscore "ark:gunzip -c $lats_dir/lat.*.gz |" ark:- | post-to-phone-post $model/final.mdl ark:- ark:- | post-to-feats --post-dim=42 ark:- ark,t:$phone_post_dir/phone_post_feat.ark || exit 1;
        set_hyphen=${set/_/-}
        input_data=$data_root_path/$set_hyphen
        output_data=$model/bnf_for_libri_light/$set/
        input_ivec_dir=$model/../ivectors_libri_light_hires/$set_hyphen
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
   for set in   dev test ; do #test; do
      for cond in  other clean ; do # other; do
	 # (
 	  echo "starting ${set}-${cond}"
          data_file=${set}-${cond}
          data_file_underscore=${data_file/-/_}
          input_data=$model/bnf_for_libri_light/$data_file_underscore
	  output_path=$model/bnf_for_libri_light_by_utt/$data_file_underscore
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
   echo "evalset: $abx_eval_set_appoint; $lm_used, $distance_mode_abx"
   source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light  # enter env: (libri-light)
   module load cuda/10.0
   PYTHONPATH=    # otherwise torchaudio cannot be imported
   echo "reset PYTHONPATH, value=$PYTHONPATH"
   libri_light_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/
   #for eval_set in dev-clean dev-other test-clean test-other; do
   #for eval_set in dev-clean ; do
   eval_set=$abx_eval_set_appoint
   eval_set_underscore=${eval_set/-/_}
   input_data=$model/bnf_for_libri_light_by_utt/
   source_dir=$input_data/${eval_set_underscore}_npy
   target_dir=$source_dir/../eval${distance_mode_abx}/${eval_set_underscore}
   mkdir -p $target_dir || exit 1;
   echo "source: $source_dir"
   #reason use python3 instead of python is that in ~/.bash_profile I add alias python=@/anaconda3/bin/python, not in @anaconda3/envs/libri-light/bin/python
   python3 ${libri_light_root}/eval/eval_ABX.py $source_dir/ ${libri_light_root}/eval/ABX_src/ABX_data/${eval_set}.item --file_extension .npy --out $target_dir/  --distance_mode $distance_mode_abx --max_size_group 50 # --cuda

   #done
   #conda deactivate
   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
   module load cuda/8.0 # reset
   #. ./path.sh
   PYTHONPAT=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH # reset
fi

#if   [ $stage -le 6 ] && [ $stop_stage -gt 6 ]; then
#  echo "construct phone posteriors using forced alignments, _NOT_ lattices "
#  input_dir=$model/align_${abx_eval_set_appoint/-/_}_nj1/ 
#  output_dir=$input_dir/phone_post/
#  mkdir -p $output_dir
#  ali-to-post "ark:gunzip -c $input_dir/ali.1.gz |" ark:- | post-to-phone-post $model/final.mdl ark:- ark:- | post-to-feats --post-dim=42 ark:- ark,t:$output_dir/feats.ark
#
#fi
#
#if   [ $stage -le 7 ] && [ $stop_stage -gt 7 ]; then
#   echo "starting ${abx_eval_set_appoint}"
#   data_file=$abx_eval_set_appoint
#   input_data=$model/align_${abx_eval_set_appoint/-/_}_nj1/phone_post
#   output_path=$input_data/data_by_utt
#   foo_path=$output_path/../
#   mkdir -p $output_path
#   cp data/$data_file/foo_template.scp $foo_path/foo_${abx_eval_set_appoint}.scp
#   sed -i "s| PathToRep| $output_path\/|g"  $foo_path/foo_${abx_eval_set_appoint}.scp || exit 1
#   copy-feats ark:$input_data/feats.ark scp,t:$foo_path/foo_${abx_eval_set_appoint}.scp || exit 1;
#   echo "${abx_eval_set_appoint}: starting removing [ and ], will construct .npy.txt files and remove .n ones"
#   if [ ! -z $output_path ] ; then
#         for file in $output_path/*.n ; do
#             sed -e "/\[/d" -e "s/\]//g" -e "s/^ //g" -e "s/ $//g"  $file  > ${file}py.txt;
#         done
#         # mkdir -p $output_path/fea_files;
#         # rm -f $output_path/*.fea || exit 1;
#         find $output_path/ -name "*.n" -delete
#         find $output_path/ -name "*.npy" -delete
#   fi
#   echo "${abx_eval_set_appoint} .npy.txt finished"
#   echo "constructing .npy format "
#   python baseline/convert_txt_npy.py $output_path
#
#
#fi
#
#if  [ $stage -le 8 ] && [ $stop_stage -gt 8 ] ; then
#   echo "$0: stage 8, alignment based phone posterior evaluated by ABX"
#   #conda activate libri-light
#   echo "enter (libri-light)"
#   echo "evalset: $abx_eval_set_appoint; $distance_mode_abx"
#   source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light  # enter env: (libri-light)
#   module load cuda/10.0
#   PYTHONPATH=    # otherwise torchaudio cannot be imported
#   echo "reset PYTHONPATH, value=$PYTHONPATH"
#   libri_light_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/
#   #for eval_set in dev-clean dev-other test-clean test-other; do
#   #for eval_set in dev-clean ; do
#   eval_set=$abx_eval_set_appoint
#   eval_set_underscore=${eval_set/-/_}
#   input_data=$model/align_${abx_eval_set_appoint/-/_}_nj1/phone_post   #$model/decode_${eval_set_underscore}_${lm_used}/phone_post_lm${lmscore} #bnf_for_libri_light/$data_file
#   source_dir=$input_data/data_by_utt_npy
##   source_dir=$model/bnf_for_libri_light_by_utt/${eval_set}_npy/
#   target_dir=$source_dir/../eval${distance_mode_abx}/
#   mkdir -p $target_dir || exit 1;
#   #reason use python3 instead of python is that in ~/.bash_profile I add alias python=@/anaconda3/bin/python, not in @anaconda3/envs/libri-light/bin/python
#   python3 ${libri_light_root}/eval/eval_ABX.py $source_dir/ ${libri_light_root}/eval/ABX_src/ABX_data/${eval_set}.item --file_extension .npy --out $target_dir/ --cuda --distance_mode $distance_mode_abx --max_size_group 50
#
#   #done
#   #conda deactivate
#   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
#   module load cuda/8.0 # reset
#   #. ./path.sh
#   PYTHONPAT=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH # reset
#
#
#fi
#
#

echo "succeeded"

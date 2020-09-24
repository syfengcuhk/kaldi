#!/bin/bash

# Siyuan Feng (2020): use a pretrained CGN chain model to decode libri-light train data and
# 	get lattices as supervision for training a chain model for libri-light
#   Assume you already finished libri-light lattice generation from a CGN chain model.
#   Or first run ../cgn_labeling.sh to get lattices

# This script was modified from the Tedlium egs.
# It assumes you first completed the run.sh from the main egs/CGN directory

## how you run this (note: this assumes that the run_tdnn.sh soft link points here;
## otherwise call it directly in its location).
# 
# local/chain/run_tdnn.sh

# This script is uses an xconfig-based mechanism
# to get the configuration.

# set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
model_name=final # resulting in final.mdl as bnf extractor; or can be 600, 700, 800, etc.

cgn_root_dir=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/cgn/s5/
# repurposed from egs/Tedlium for CGN by LvdW
remove_egs=false
stop_stage=19
stage=17
align_fmllr_lats_stage=-10
nj=1
num_threads_decode=1
decode_nj=4
min_seg_len=1.55
xent_regularize=0.1
cgn_train_set=train
train_set=train_unlab_600
train_subset=_subset3600utt
lat_generator_acwt=10.0
gmm=tri4  # the gmm for the target data
num_threads_ubm=32
nnet3_affix=  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned


abx_eval_set_appoint=dev-clean
# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
train_exit_stage=0
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=1a  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
rel_train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_libri_light_hires/${train_set}

# some settings dependent on the GPU, for a single GTX980Ti these settings seem to work ok.
#
# increase these if you have multiple GPUs
num_jobs_initial=1
num_jobs_final=1
num_epochs=5
# change these for different amounts of memory
#num_chunks_per_minibatch="256,128,64"
num_chunks_per_minibatch="128,64"
frames_per_iter=1500000
lat_generator_model=$cgn_root_dir/exp/chain/bn_layer/tdnn1a_sp_bi_epoch3_inputmfcc_hires_only
# resulting in around 2500 iters
# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# local/nnet3/run_ivector_common.sh --stage $stage \
#                                   --nj $nj \
#                                   --min-seg-len $min_seg_len \
#                                   --train-set $train_set \
#                                   --gmm $gmm \
#                                   --num-threads-ubm $num_threads_ubm \
#                                   --nnet3-affix "$nnet3_affix"


gmm_dir=$cgn_root_dir/exp/$cgn_train_set/$gmm
ali_dir=exp/chain${nnet3_affix}_cgn_lat_label/${train_set}${train_subset}/lat_gen_acwt${lat_generator_acwt}/${train_set}${train_subset}_ali

tree_dir=exp/chain${nnet3_affix}_cgn_lat_label/${train_set}${train_subset}/lat_gen_acwt${lat_generator_acwt}/tree_bi${tree_affix}
#lat_dir=$lat_generator_model/decoding_for_libri_light_${train_set}${train_subset}_acwt${lat_generator_acwt}
lat_dir=exp/chain${nnet3_affix}_cgn_lat_label/${train_set}${train_subset}/lat_gen_acwt${lat_generator_acwt}/${train_set}${train_subset}_lats
#if [ ! -f $lat_dir/final.mdl ]; then
#  cp $lat_dir/../final.mdl $lat_dir/.
#fi

input_feat_affix=mfcc_hires_only
dir=exp/chain${nnet3_affix}_cgn_lat_label/${train_set}${train_subset}/lat_gen_acwt${lat_generator_acwt}/dnn${tdnn_affix}_bi_epoch${num_epochs}_input${input_feat_affix}
train_data_dir=data_hires_conf_cgn/${train_set}${train_subset}

lores_train_data_dir=data_plus_cgn_transcripts/acwt${lat_generator_acwt}/${train_set}${train_subset}
common_egs_dir=exp/chain_cgn_lat_label/${train_set}${train_subset}/lat_gen_acwt${lat_generator_acwt}/tdnn1a_bi_epoch5_inputmfcc_hires_only/egs

for f in  $train_data_dir/feats.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

#if [ $stage -le 14 ]; then
#  echo "$0: creating lang directory with one state per phone."
#  # Create a version of the lang/ directory that has one state per phone in the
#  # topo file. [note, it really has two states.. the first one is only repeated
#  # once, the second one has zero or more repeats.]
#  if [ -d data/lang_chain ]; then
#    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
#      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
#    else
#      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
#      echo " ... not sure what to do.  Exiting."
#      exit 1;
#    fi
#  else
#    cp -r data/lang_s data/lang_chain
#    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
#    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
#    # Use our special topology... note that later on may have to tune this
#    # topology.
#    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
#  fi
#fi

if [ $stage -le 15 ] && [  $stop_stage -gt 15 ]  ; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --stage $align_fmllr_lats_stage --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    $cgn_root_dir/data/lang_s $gmm_dir $lat_dir
  #rm $lat_dir/fsts.*.gz # save space
fi
if [ $stage -le 16 ] && [ $stop_stage -gt 16 ] ; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} $cgn_root_dir/data/lang_chain $ali_dir $tree_dir
fi

if [ $stage -le 17 ] && [ $stop_stage -gt 17 ]  ; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=450 self-repair-scale=1.0e-04
  relu-batchnorm-layer name=tdnn2 dim=450
  relu-batchnorm-layer name=tdnn3 dim=450
  relu-batchnorm-layer name=tdnn4 dim=450
  relu-batchnorm-layer name=tdnn5 dim=450
  relu-batchnorm-layer name=tdnn6 dim=40

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=450 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=450 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

 

if [ $stage -le 18 ] && [ $stop_stage -gt 18 ] ; then
 echo "$0: hostname: $(hostname)"
 steps/nnet3/chain/train.py --stage $train_stage --exit-stage $train_exit_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --feat.online-ivector-dir $cgn_root_dir/$rel_train_ivector_dir \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch $num_chunks_per_minibatch \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --use-gpu "wait" \
    --dir $dir   || exit 1;
fi

#We dont need to decode libri-light with the above model

# if [ $stage -le 19 ]; then
#   # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
#   # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
#   # the lang directory.
#   utils/mkgraph.sh --self-loop-scale 1.0 data/lang_s_test_tgpr $dir $dir/graph
# fi

# if [ $stage -le 20 ]; then
#   for x in dev_s dev_t_16khz; do
#     nspk=$(wc -l <data/$x/spk2utt)
#     [ "$nspk" -gt "$decode_nj" ] && nspk=$decode_nj
#     steps/nnet3/decode.sh --num-threads $num_threads_decode --nj $nspk --cmd "$decode_cmd" \
#       --acwt 1.0 --post-decode-acwt 10.0 \
#       --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${x}_hires \
#       --scoring-opts "--min-lmwt 5 " \
#       $dir/graph data/${x}_hires $dir/decode_${x} || exit 1;
#     steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_s_test_{tgpr,fgconst} \
#       data/${x}_hires ${dir}/decode_${x} ${dir}/decode_${x}_rescore || exit
#   done
# fi
echo "finished chain model..."

if [ $stage -le 21 ] && [ $stop_stage -gt 21 ];then
  echo "extract BNF of libri-light dev-clean ~ test-other"
  bnf_name=tdnn6.affine
  data_root_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/data_hires_conf_cgn

  for set in  dev-other dev-clean test-other test-clean; do
    if [ "$model_name" = "final" ]; then
      output_data=$dir/bnf_for_libri_light/$set/
    else # use intermediate model *.mdl as extractor, specify it
      echo "using intermediate model ${model_name}.mdl"
      output_data=$dir/bnf_for_libri_light_${model_name}/$set
    fi
    input_data=$data_root_path/$set
#    output_data=$dir/bnf_for_libri_light/$set
    input_ivec_dir=$cgn_root_dir/exp/nnet3/ivectors_libri_light_hires/$set
    steps/nnet3/make_bottleneck_features.sh --use-gpu true --model-name $model_name \
           --cmd run.pl  --nj $nj --ivector-dir $input_ivec_dir \
           $bnf_name $input_data $output_data $dir || exit 1;
  done

fi

if [ $stage -le 22 ] && [ $stop_stage -gt 22 ];then
   for set in dev test; do
      for cond in clean other; do
         # (
          echo "starting ${set}-${cond}"
          data_file=${set}-${cond}
          if [ "$model_name" = "final" ]; then
            input_data=$dir/bnf_for_libri_light/$data_file
            output_path=$dir/bnf_for_libri_light_by_utt/$data_file
          else
            input_data=$dir/bnf_for_libri_light_${model_name}/$data_file
            output_path=$dir/bnf_for_libri_light_by_utt_${model_name}/$data_file
          fi

#          input_data=$dir/bnf_for_libri_light/$data_file
#          output_path=$dir/bnf_for_libri_light_by_utt/$data_file
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

if   [ $stage -le 23 ] && [ $stop_stage -gt 23 ]; then
   echo "$0: stage 23, evaluated by ABX"
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
   if [ "$model_name" = "final" ];then
     source_dir=$dir/bnf_for_libri_light_by_utt/${eval_set}_npy/
     target_dir=$dir/bnf_for_libri_light_by_utt/eval/$eval_set
   else
     source_dir=$dir/bnf_for_libri_light_by_utt_${model_name}/${eval_set}_npy/
     target_dir=$dir/bnf_for_libri_light_by_utt_${model_name}/eval/$eval_set
   fi
#   source_dir=$dir/bnf_for_libri_light_by_utt/${eval_set}_npy/
#   target_dir=$dir/bnf_for_libri_light_by_utt/eval/$eval_set
   mkdir -p $target_dir || exit 1;
   #reason use python3 instead of python is that in ~/.bash_profile I add alias python=@/anaconda3/bin/python, not in @anaconda3/envs/libri-light/bin/python
   echo "Exp. dir: $target_dir"
   python3 ${libri_light_root}/eval/eval_ABX.py $source_dir/ ${libri_light_root}/eval/ABX_src/ABX_data/${eval_set}.item --file_extension .npy --out $target_dir/ --cuda

   #done
   #conda deactivate
   source /scratch/siyuanfeng/software/anaconda3/bin/deactivate # reset
   module load cuda/8.0 # reset
   #. ./path.sh
   PYTHONPAT=./:./src:$KALDI_PYTHON_DIR:$PYTHONPATH # reset
fi

exit 0

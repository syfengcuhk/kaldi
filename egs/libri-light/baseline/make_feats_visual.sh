#!/bin/bash
nj=1
stage=0
stop_stage=1
root_dir=data_for_visualization
feat_type=mfcc
target_phone=AE
. ./utils/parse_options.sh
source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light
for set in dev-clean ; do
  target_dir=$root_dir/$feat_type/${set}_$target_phone
  if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
      for set in  dev-other dev-clean test-other test-clean; do
          steps/make_mfcc.sh --nj $nj $target_dir || exit 1;
          steps/compute_cmvn_stats.sh $target_dir || exit 1;
      done
  fi
done
source /scratch/siyuanfeng/software/anaconda3/bin/deactivate

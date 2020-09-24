#!/bin/bash
nj=1
stage=4
stop_stage=5
hires_conf_set=cgn # or hkust, or aidatatang
data_dir_name=data_hires_conf_${hires_conf_set}
. ./utils/parse_options.sh
. ./path.sh
echo "S0: enter virtual env: (libri-light)"
source /scratch/siyuanfeng/software/anaconda3/bin/activate libri-light
if [ $stage -le 0 ] && [ $stop_stage -gt 0 ];then
    for set in  dev-other dev-clean test-other test-clean; do
        utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
        steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
        steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -gt 1 ];then
   for set in train_unlab_600 unseg_train_unlab_600 ; do
       utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
       steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
       steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
   done
fi

if [ $stage -le 2 ] && [ $stop_stage -gt 2 ];then
   for set in train_unlab_6k  ; do
       utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
       steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
       steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
   done
fi

if [ $stage -le 3 ] && [ $stop_stage -gt 3 ];then
   for set in dev test; do
      for cond in clean other; do
	 # (
 	  echo "starting ${set}-${cond}"
          data_file=${set}-${cond}
	  output_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/${data_dir_name}_by_utt/$data_file
          foo_path=$output_path/../
          mkdir -p $output_path
	  cp data/$data_file/foo_template.scp $foo_path/foo_${set}-${cond}.scp
   	  sed -i "s| PathToRep| $output_path\/|g"  $foo_path/foo_${set}-${cond}.scp || exit 1
          #copy-feats scp:data/$data_file/feats.scp - | 
	  apply-cmvn --utt2spk=ark:${data_dir_name}/$data_file/utt2spk scp:${data_dir_name}/$data_file/cmvn.scp scp:${data_dir_name}/$data_file/feats.scp scp,t:$foo_path/foo_${set}-${cond}.scp || exit 1;
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

if [ $stage -le 4 ] && [ $stop_stage -gt 4 ];then
   for set in train_unlab_600_subset3600utt train_unlab_600_subset7200utt train_unlab_600_subset14400utt ; do
       utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
       steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
       steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
   done
fi
if [ $stage -le 5 ] && [ $stop_stage -gt 5 ];then
   for set in train_unlab_600_subset900utt  ; do
       utils/copy_data_dir.sh data/$set ${data_dir_name}/$set || exit 1;
       steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_from_${hires_conf_set}.conf --nj $nj ${data_dir_name}/$set || exit 1;
       steps/compute_cmvn_stats.sh ${data_dir_name}/$set || exit 1;
   done
fi

source deactivate 
echo "succeeded"

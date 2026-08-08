#!/bin/bash
# Usage: run_checks.sh <epoch-tag e.g. 00010>
EP=$1
module load apptainer/apptainer.module
WORKSPACE=/home/sk1064/workspace
OUT=$WORKSPACE/EchoFM_gh/logs/${ECHOFM_RUN_TAG:-echofm_mae_vitl_f32_v3_0807}
srun -p defq --gres=gpu:a100-40:1 --cpus-per-task=8 -t 40 -J "mm_fm_training::proj=IRB2025P001686," \
  apptainer exec --nv --writable-tmpfs \
  --bind $WORKSPACE:$WORKSPACE \
  --bind /mnt/weka/wekafs/mm_fm_training:/mnt/weka/wekafs/mm_fm_training \
  $WORKSPACE/lvfp/sk1064_lvfp.sif bash -lc "
    cd $WORKSPACE/EchoFM_gh
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    export PYTHONPATH=$WORKSPACE/lvfp/extra_pkgs:\$PYTHONPATH
    D=/mnt/weka/wekafs/mm_fm_training/echo_pretrain_apical2/clips
    python tools/check_periodicity.py --ckpt $OUT/checkpoint-$EP.pth \
      --data_path \$D --num_clips 32 --out $OUT/periodicity_ep$EP
    python tools/check_recon.py --ckpt $OUT/checkpoint-$EP.pth \
      --data_path \$D --num_clips 16 --out $OUT/recon_ep$EP
  "

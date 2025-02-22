NUM_GPUS=8

for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python eval_realstate.py \
  --seed 12345 \
  --processed_video_dir /homes/55/runjia/storage/realestate10k/video_data_processed_sd21 \
  --meta_file /homes/55/runjia/storage/realestate10k/RealEstate10K \
  --ckpt ./checkpoints/motionctrl_svd.ckpt \
  --config ./configs/inference/config_motionctrl_cmcm.yaml \
  --savedir /homes/55/runjia/storage/realestate10k_results/motionctrl \
  --savefps 10 \
  --ddim_steps 25 \
  --frames 14 \
  --input ./examples/basic/eduardo-gorghetto-5auIBbcoRNw-unsplash.jpg \
  --fps 10 \
  --motion 127 \
  --cond_aug 0.02 \
  --decoding_t 1 \
  --resize \
  --height 576 \
  --width 1024 \
  --sample_num 2 \
  --transform \
  --pose_dir ./examples/camera_poses \
  --speed 1.0 \
  --num_gpus $NUM_GPUS \
  --rank $i &
done
wait
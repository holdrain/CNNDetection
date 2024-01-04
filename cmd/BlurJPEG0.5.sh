#!/bin/bash
accelerate launch --num_processes 4 --main_process_port 29501 train.py --gpu_ids 0,1,2,3 --name blur_jpg_prob0.5 --blur_prob 0.5 --blur_sig 0.0,3.0 --jpg_prob 0.5 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /mnt/shared/cnndetection2020  \
--classes airplane,bird,bicycle,boat,bottle,bus,car,cat,cow,chair,diningtable,dog,person,pottedplant,motorbike,tvmonitor,train,sheep,sofa,horse \
--batch_size 800 --wandb_id 0005
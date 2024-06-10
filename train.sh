export CUDA_VISIBLE_DEVICES=0,1,2,3;
python train.py --name df3_faces --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --dataroot /DATAROOT/ --batch_size 64 --lr 0.00005 --gpu_ids 0,1,2,3 --classes model1,model2, 


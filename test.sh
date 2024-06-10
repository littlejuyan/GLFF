export CUDA_VISIBLE_DEVICES=7;

python eval.py --blur_prob 0 --blur_sig 4 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 20 

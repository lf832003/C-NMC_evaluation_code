#!/bin/bash
FILES=$1

echo Preprocessing image!
mkdir ../input_images_tmp
python utils.py --image $FILES

echo Predicting by Model-A!
mkdir ../prediction
python predict.py --model_path ../model/model_1.h5 --test_path ../input_images_tmp --save_npy ../prediction/model_1.npy 

echo Predicting by Model-B!
python predict.py --model_path ../model/model_2.h5 --test_path ../input_images_tmp --save_npy ../prediction/model_2.npy

echo Predicting by combined model!
python predict.py --model_path ../model/weights-07-0.93.hdf5 --test_path ../input_images_tmp --save_npy ../prediction/model_combined.npy

echo Making inference!
python inference.py --prediction_path ../prediction --save_pkl ../Pred.pkl --mode 2
python process_pkl.py --pkl_path ../Pred.pkl

rm -r ../input_images_tmp
rm -r ../prediction
rm ../Pred.pkl

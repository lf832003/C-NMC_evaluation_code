import os
import pickle
import argparse

import numpy as np

from IO import readfileslist

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_path', '-p', help = 'Folder containing the npy file')
parser.add_argument('--save_pkl', '-s')
parser.add_argument('--mode', '-m')
args = parser.parse_args()

def inference(prediction_list, confidence = 0.6, mode = 0):
    prediction = []
    prediction_1 = np.load(prediction_list[0])
    prediction_2 = np.load(prediction_list[1])
    prediction_fine_tune = np.load(prediction_list[2])

    if mode == 0:
        for idx in xrange(len(prediction_1)):
            maxidx_1 = np.argmax(prediction_1[idx, :])
            maxidx_2 = np.argmax(prediction_2[idx, :])

            if maxidx_1 != maxidx_2: # or prediction_1[idx, maxidx_1] < confidence or prediction_2[idx, maxidx_2] < confidence:
                if np.argmax(prediction_fine_tune[idx, :]):
                    prediction.append('hem')
                else:
                    prediction.append('all')
                # continue
            else:
                if maxidx_1 == 0:
                    prediction.append('all')
                else:
                    prediction.append('hem')

            # if prediction_1[idx, 0] + prediction_2[idx, 0] >= prediction_1[idx, 1] + prediction_2[idx, 1]:
            #     prediction.append('all')
            # else:
            #     prediction.append('hem')
    elif mode == 1:
        for idx in xrange(len(prediction_1)):
            pred_all = prediction_1[idx, 0] + prediction_2[idx, 0] + prediction_fine_tune[idx, 0]
            pred_hem = prediction_1[idx, 1] + prediction_2[idx, 1] + prediction_fine_tune[idx, 1]

            if pred_all >= pred_hem:
                prediction.append('all')
            else:
                prediction.append('hem')
    elif mode == 2:
        for idx in xrange(len(prediction_1)):
            pred = np.concatenate((prediction_1[idx, :], prediction_2[idx, :], prediction_fine_tune[idx, :]))
            pred_idx = np.argmax(pred)

            if pred_idx == 0 or pred_idx == 2 or pred_idx == 4:
                prediction.append('all')
            else:
                prediction.append('hem')

    elif mode == 3:
        for idx in xrange(len(prediction_1)):
            pred_all = prediction_1[idx, 0] + prediction_2[idx, 0]
            pred_hem = prediction_1[idx, 1] + prediction_2[idx, 1]

            if pred_all >= pred_hem:
                prediction.append('all')
            else:
                prediction.append('hem')
    elif mode == 4:
        for idx in xrange(len(prediction_1)):
            pred = np.concatenate((prediction_1[idx, :], prediction_2[idx, :]))
            pred_idx = np.argmax(pred)

            if pred_idx == 0 or pred_idx == 2:
                prediction.append('all')
            else:
                prediction.append('hem')
    elif mode == 5:
        for idx in xrange(len(prediction_1)):
            if prediction_fine_tune[idx, 0] >= prediction_fine_tune[idx, 1]:
                prediction.append('all')
            else:
                prediction.append('hem')

    return prediction

if __name__ == '__main__':
    args = parser.parse_args()
    npy_list = readfileslist(args.prediction_path, '.npy')
    prediction = inference(npy_list, mode = int(args.mode))

    with open(args.save_pkl, 'w') as fid:
        pickle.dump(prediction, fid)        

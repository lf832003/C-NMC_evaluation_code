import os
import pickle
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', '-p', help = 'path to Pred.pkl')
args = parser.parse_args()

def main(pred_pkl):    
    with open(pred_pkl, 'r') as fid:
        pred = pickle.load(fid)
    
    pred_arr = np.zeros(len(pred), dtype = np.uint8)
    
    with open('../isbi_valid.predict', 'w') as fid:
        for i in xrange(len(pred)):
            if pred[i] == 'all':
                fid.write('1\n')
                pred_arr[i] = 1
            elif pred[i] == 'hem':
                fid.write('0\n')
    pd.DataFrame(pred_arr).to_csv('../Pred.csv', header = None, index = None)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.pkl_path)

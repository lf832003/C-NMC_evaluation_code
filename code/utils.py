import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import OrderedDict
from skimage import img_as_ubyte
from skimage.color import rgb2gray, separate_stains, bex_from_rgb
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk
from skimage.transform import resize

from imgaug import augmenters as iaa
from IO import readfileslist

parser = argparse.ArgumentParser()
parser.add_argument('--image', required = True, type = str, help = 'Path to the image folder')
parser.add_argument('--output', default = '../input_images_tmp', type = str, help = 'Path to the output folder')

def im2bw(img):
    thresh = threshold_otsu(img)
    return (img > thresh).astype(np.uint8)

def single_patch_extract(img, rect, target_shape = [224, 224]):
    img_crop = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
    return img_as_ubyte(resize(img_crop, target_shape)) 

def img_normalization(img):
    return np.clip((img.astype(np.float32) - np.min(img)) / (np.max(img) - np.min(img)), 0.0, 1.0)

def rgb2stain(img, diameter = 3):
    mask = im2bw(rgb2gray(img))
    mask = binary_erosion(mask, disk(diameter)).astype(np.uint8)
    
    img_t = separate_stains(img, bex_from_rgb)
    img_t = img_as_ubyte(img_normalization(img_t))
    
    return img_t * np.expand_dims(mask, axis = 2)

def preprocess(img, target_shape):
    img_gray = rgb2gray(img)
    img_binary = im2bw(img_gray)
    img_label = label(img_binary)
    props = regionprops(img_label)
    if len(props) > 1:
        idx = np.argmax([prop.area for prop in props])
    else:
        idx = 0
    return single_patch_extract(img, props[idx].bbox, target_shape)

def create_dataframe(imgnamelist):
    filename = []
    filepath = []
    patientID = []
    imageNumber = []
    cellID = []
    category = []
    
    for imgname in imgnamelist:
        file_name = os.path.basename(imgname)
        filename.append(file_name)
        filepath.append(imgname)
        _idx = [pos for pos, val in enumerate(file_name) if val == '_']
        patientID.append(file_name[_idx[0] + 1:_idx[1]])
        imageNumber.append(file_name[_idx[1] + 1:_idx[2]])
        cellID.append(file_name[_idx[2] + 1:_idx[3]])
        category.append(file_name[_idx[3] + 1:len(os.path.basename(imgname)[0:-4])])
    
    df = pd.DataFrame(OrderedDict((('FileName', filename), ('Patient ID', patientID), ('Image ID', imageNumber), 
                                   ('Cell ID', cellID), ('Category', category), ('FilePath', filepath))))
    
    return df

def create_augmenters():
    seq = iaa.Sequential(iaa.SomeOf((1, 4), [
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)), 
            iaa.AverageBlur(k = (1, 5)), 
            iaa.MedianBlur(k = (1, 5)), 
        ]),
        iaa.AdditiveGaussianNoise(loc = 0, scale = (0.0, 0.03 * 255), per_channel = 0.5), 
        iaa.AddToHueAndSaturation((-1, 1)), 
        iaa.ContrastNormalization((0.5, 1.0), per_channel = 0.5), 
    ], random_order = True))
    return seq

def main(argv):
    bmplist = readfileslist(FLAGS.image, '.bmp')
    for bmp in bmplist:
        img = imread(bmp)
        img = preprocess(img, (299, 299))
        imsave(os.path.join(FLAGS.output, os.path.basename(bmp)), img)

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = sys.argv[:1] + unparsed)

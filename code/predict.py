import os
import pickle

import tensorflow as tf
import numpy as np

from IO import readfileslist
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_string('model_path', '', 'Path to saved model (.h5)')
flags.DEFINE_string('test_path', '', 'Path to test folder')
flags.DEFINE_string('save_npy', '', 'Path to save numpy array file')
flags.DEFINE_string('save_pkl', '', 'Path to save y_true (.pkl)')
flags.DEFINE_string('save_labels', '../Cell_data/isbi_valid.predict', 'Path to save labels (.predict)')
flags.DEFINE_string('filenames', '', 'Path to save test generator filenames (.pkl)')
FLAGS = flags.FLAGS

def model_predict_1(model_path, test_path):
    model = load_model(model_path)
    imglist = readfileslist(test_path, '.bmp')
    
    img_mat = np.zeros([len(imglist), 299, 299, 3], dtype = np.float32)

    filenames = []
    for idx, imgname in enumerate(imglist):
        filenames.append(os.path.basename(imgname))
        img_mat[idx] = imread(imgname) / 255.0

    pred = model.predict(img_mat)

    # print pred.shape
    classes_numerical = np.argmax(pred, axis = 1)

    return pred, 1 - classes_numerical, filenames  

def model_predict(model_path, test_path, test_mode = 0):
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

    test_generator = test_datagen.flow_from_directory(test_path, target_size = (299, 299), 
                                                      shuffle = False, batch_size = 1)
    if test_mode == 0:
        y_true = []
        for file_name in test_generator.filenames:
            file_basename = os.path.basename(file_name)[0:-4]
            _idx = [pos for pos, s in enumerate(file_basename) if s == '_']        
            y_true.append(file_basename[_idx[3] + 1:_idx[3] + 4])           

        return model.predict_generator(test_generator, steps = len(test_generator.filenames)), y_true, test_generator.filenames
    elif test_mode == 1:
        
        return model.predict_generator(test_generator, steps = len(test_generator.filenames)), test_generator.filenames

def main(_):
    if FLAGS.save_pkl != '':
        predict, y_true, filenames = model_predict(FLAGS.model_path, FLAGS.test_path)
        with open(FLAGS.save_pkl, 'w') as fid:
            pickle.dump(y_true, fid)
    else:
        # predict, filenames = model_predict(FLAGS.model_path, FLAGS.test_path, test_mode = 1)
        predict, labels, filenames = model_predict_1(FLAGS.model_path, FLAGS.test_path)

    np.save(FLAGS.save_npy, predict)
    # np.save(FLAGS.save_labels, labels)
#    with open(FLAGS.save_labels, 'w') as fid:
#        for i in xrange(len(labels)):
#            fid.write(str(labels[i]) + '\n')

    # with open(FLAGS.filenames, 'w') as fid:
    #     pickle.dump(filenames, fid)

    return None

if __name__ == '__main__':
    tf.app.run()

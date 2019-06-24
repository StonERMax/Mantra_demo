import os

import sys
sys.path.insert(0,'lib/')
sys.path.insert(0, 'seq/')
sys.path.insert(0,'/nas/medifor/yue_wu/thirdparty/DeepNonLocalMeansWithPriors/lib')
import numpy as np 
import cv2 as cv2
import lmdb
import keras
from matplotlib import pyplot
np.set_printoptions( 3, suppress = True )
from tensorflow.python.client import device_lib
from tf_multi_gpu import make_parallel 
print device_lib.list_local_devices()
from keras.utils import to_categorical
import modelCore

print cv2.__version__
fontsize = 24

use_model = "FF0_RR1_RC1_noABS"
debug = False   #

####################################################################################################
# Prepare Model and Metrics
####################################################################################################
import sys
sys.path.insert( 0, 'sequence/')


manTraNet_root = './'
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )

def get_single_gpu_model( use_model ) :
    
    mdel = modelCore.load_pretrain_model_by_index( 1, manTraNet_modelDir )
    print mdel.summary(line_length=120)
    
    return mdel

from sklearn.metrics import roc_auc_score, f1_score
import tensorflow as tf

def np_F1( y_true, y_pred ) :
    score = []
    for yy_true, yy_pred in zip( y_true, y_pred ) :
        this = f1_score( (yy_true>.5).astype('int').ravel(), (yy_pred>.5).astype('int').ravel() )
        that = f1_score( (yy_true>.5).astype('int').ravel(), (1-yy_pred>.5).astype('int').ravel() )
        score.append( max( this, that ) )
    return np.mean( score ).astype('float32')


def F1( y_true, y_pred ) :
    return tf.py_func( np_F1, [y_true, y_pred], 'float32')

def np_auc( y_true, y_pred ) :
    score = []

    for yy_true, yy_pred in zip( y_true, y_pred ) :
        this = roc_auc_score( (yy_true>.5).astype('int').ravel(), yy_pred.ravel() )
        that = roc_auc_score( (yy_true>.5).astype('int').ravel(), 1-yy_pred.ravel() )
        score.append( max( this, that ) )
    return np.mean( score ).astype('float32')

def auroc(y_true, y_pred):
    return tf.py_func( np_auc, [y_true, y_pred], 'float32')

####################################################################################################
# Prepare Testing Dataset
####################################################################################################
def prepare_coverage_dataset() :
    input_image_file_list = 'COVERAGE_image_new.list'
    with open( input_image_file_list, 'r') as IN :
        input_files = [ line.strip() for line in IN.readlines() ]
    print "INFO: successfully load", len( input_files ), "input files"

    def get_input_ID( input_file ) :
        bname = os.path.basename( input_file )
        return bname.rsplit('.')[0]

    def get_mask_file_from_ID( sample_id ) :
        return os.path.join('VERAGE/mask/', '{}forged.tif'.format(sample_id[:-1]) ) 

    def preprocess( input_image, input_mask ) :
        x = np.expand_dims( input_image, axis=0 ).astype('float32')/255. * 2 - 1
        y = np.expand_dims( np.expand_dims( input_mask, axis=0 ), axis=-1 )/255.
        return x, y

    raw_lut = dict( zip( [ get_input_ID(f) for f in input_files ], input_files) )

    paired_results = []
    c=0
    for key in raw_lut.keys() : 
        c = c+1
        raw_file = raw_lut[key]
        mask_file = get_mask_file_from_ID(key)
        
        r = cv2.imread( raw_file, 1 )[...,::-1]
        m = cv2.imread( mask_file, 0)
        if r.shape[:2] != m.shape[:2] :
            continue
        
        raw_mask_dec = ( raw_file, mask_file )
        paired_results.append( raw_mask_dec )

    #print(c)
    print len(paired_results)
    return paired_results, len(paired_results), preprocess

def prepare_columbia_dataset() :
    input_image_file_list = 'sequence/Columbia_image_new.list'
    with open( input_image_file_list, 'r') as IN :
        input_files = [ line.strip() for line in IN.readlines() ]
    print "INFO: successfully load", len( input_files ), "input files"

    def get_input_ID( input_file ) :
        bname = os.path.basename( input_file )
        return bname.rsplit('.')[0]

    def get_mask_file_from_ID( sample_id ) :
        return os.path.join('Columbia/mixed/edgemask/', '{}_edgemask.jpg'.format(sample_id) ) 

    def preprocess( input_image, input_mask ) :
        x = np.expand_dims( input_image, axis=0 ).astype('float32')/255. * 2 - 1
        y = np.expand_dims( np.expand_dims( input_mask, axis=0 ), axis=-1 )/255.
        return x, y

    raw_lut = dict( zip( [ get_input_ID(f) for f in input_files ], input_files) )

    paired_results = []
    for key in raw_lut.keys() : 
        raw_file = raw_lut[key]
        mask_file = get_mask_file_from_ID(key)
        
        #print raw_file
        #print mask_file

        r = cv2.imread( raw_file, 1 )[...,::-1]
        m = cv2.imread( mask_file, 0)
        if r.shape[:2] != m.shape[:2] :
            continue

        raw_mask_dec = ( raw_file, mask_file )
        paired_results.append( raw_mask_dec )

    print len(paired_results)
    return paired_results, len(paired_results), preprocess

def prepare_nist_dataset( check=True ) :
    input_image_file_list = 'NIST16/mani_new.list'
    with open( input_image_file_list, 'r') as IN :
        input_files = [ line.strip().split( ) for line in IN.readlines() ]
    print "INFO: successfully load", len( input_files ), "input files"
    def preprocess( input_image, input_mask ) :
        h, w = input_image.shape[:2]
        r = 1024./min(h,w)
        nh, nw = int(h*r), int(w*r)
        x = np.expand_dims( cv2.resize(input_image, (nw,nh), interpolation=cv2.INTER_LINEAR), axis=0 ).astype('float32')/255. * 2 - 1
        y = np.expand_dims( np.expand_dims( cv2.resize(1-input_mask/255, (nw,nh), interpolation=cv2.INTER_NEAREST), axis=0 ), axis=-1 )
        return x, y
    paired_results = []
    for raw_file, mask_file in input_files :
        #print os.path.isfile('NIST16/'+raw_file) and os.path.isfile('NIST16/'+mask_file)
        if check :
            r = cv2.imread( 'NIST16/'+raw_file, 1 )[...,::-1]
            m = cv2.imread( 'NIST16/'+mask_file, 0)
            if r.shape[:2] != m.shape[:2] :
                continue
        
        pyplot.figure( figsize=(15,5) )
        pyplot.subplot(121)
        pyplot.imshow( r )
        pyplot.subplot(122)
        pyplot.imshow( m, cmap='gray' )
        pyplot.show()


        raw_mask_dec = ( raw_file, mask_file )
        paired_results.append( raw_mask_dec )

    print len(paired_results)
    return paired_results, len(paired_results), preprocess

def prepare_casia_dataset() :
    input_image_file_list = 'sequence/CASIA_image.list'
    with open( input_image_file_list, 'r') as IN :
        input_files = [ line.strip() for line in IN.readlines() ]
    print "INFO: successfully load", len( input_files ), "input files"

    def get_input_ID( input_file ) :
        bname = os.path.basename( input_file )
        return bname.rsplit('.')[0]

    #def get_mask_file_from_ID( sample_id ) :
    #    return os.path.join('/nas/vista-ssd01/medifor/yue_wu/Dataset/CameraModel/Cleaned/CASIAY1/', '{}_mask.bmp'.format(sample_id) )   ### NOT PRESENT

    def preprocess( input_image, input_mask ) :
        x = np.expand_dims( input_image, axis=0 ).astype('float32')/255. * 2 - 1
        y = np.expand_dims( np.expand_dims( input_mask, axis=0 ), axis=-1 )/255.
        return x, y

    raw_lut = dict( zip( [ get_input_ID(f) for f in input_files ], input_files) )

    paired_results = []
    for key in raw_lut.keys() : 
        raw_file = raw_lut[key]
        mask_file = get_mask_file_from_ID(key)
        r = cv2.imread( raw_file, 1 )[...,::-1]
        m = cv2.imread( mask_file, 0)
        if r.shape[:2] != m.shape[:2] :
            continue
        raw_mask_dec = ( raw_file, mask_file )
        paired_results.append( raw_mask_dec )

    print len(paired_results)
    return paired_results, len(paired_results), preprocess

####################################################################################################
# Evaluate model performance for each dataset
####################################################################################################
def create_evaluation_data_generator( paired_results, preprocess ) :
    for raw_file, mask_file in paired_results :
        r = cv2.imread( raw_file, 1 )[...,::-1]
        m = cv2.imread( mask_file, 0)
        if r.shape[:2] != m.shape[:2] :
            print "INFO: find unmatched", raw_file, mask_file, ", skip"
            continue
        x, y = preprocess( r, m )
        yield x, y


model = get_single_gpu_model( use_model )
#model = keras.utils.multi_gpu_model(model,2)
model.compile( optimizer='sgd',
               loss ='binary_crossentropy',
               metrics=['accuracy', F1, auroc], )


from prettytable import PrettyTable
import os
import json

table = PrettyTable()
table.field_names = ['Dataset', 'Loss', 'Acc/AUC', 'F1' ]

mega_lut = dict()
'''
for prepare_dataset, name in zip( [ prepare_casia_dataset,
                                    prepare_nist_dataset,
                                    prepare_columbia_dataset,
                                    prepare_coverage_dataset, ],
                                  ['CASIA', 'NIST', 'COLUMBIA', 'COVERAGE'] ) :
'''
for prepare_dataset, name in zip( [ prepare_nist_dataset ],
                                  ['NIST'] ) :
    input_pairs, L, preprocess = prepare_dataset()
    # create data generator
    datagen = create_evaluation_data_generator( input_pairs, preprocess ) 
    res = model.evaluate_generator( datagen, L if not debug else 1, verbose=1 )
    # print 
    lut = dict( zip( model.metrics_names, res ) )
    print name, lut
    mega_lut[name] = lut
    # update
    table.add_row( [name] + [ "{:.4f}".format(lut[key]) for key in ['loss', 'auroc', 'F1'] ] )

print (table)



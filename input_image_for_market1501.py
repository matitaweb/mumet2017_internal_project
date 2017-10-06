import numpy as np
np.random.seed(1217)
import h5py
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.preprocessing import image as pre_image
from make_hdf5_for_market1501 import get_image_path_list, get_list_positive_index_market1501



class NumpyArrayIterator_for_Market1501(pre_image.Iterator):
    
    def __init__(self, f, path_list, train_or_validation = 'train', flag = 0, image_data_generator = None, batch_size=32, shuffle=False, seed=None, dim_ordering='default'):
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.f = f
        self.path_list = path_list
        self.folder_dir = 'dataset/market-1501/bounding_box_' + train_or_validation + '/'
        self.train_or_validation = train_or_validation
        self.flag = flag
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.train_or_validation_n = 1
        if(train_or_validation == 'train'):
            self.train_or_validation_n = 0
        print('indice: ' + str(self.train_or_validation_n) + " -> " + str(f[self.train_or_validation_n].shape))
        super(NumpyArrayIterator_for_Market1501, self).__init__(f[self.train_or_validation_n].shape[0], batch_size, shuffle, seed)

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            print (' shape: '+str(index_array.shape) + ', current_index: ' + str(current_index) + ', current_batch_size: ' + str(current_batch_size))
            
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)
                   
    def next(self):
        with self.lock:
            #print(self.index_generator)
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        batch_x1 = np.zeros(tuple([current_batch_size * 2] + [128,64,3]))
        batch_x2 = np.zeros(tuple([current_batch_size * 2] + [128,64,3]))
        batch_y  = np.zeros([current_batch_size * 2, 2])
        for i, j in enumerate(index_array):
            x1 = np.array(Image.open(self.folder_dir + self.f[self.train_or_validation_n][j,0])) / 255.
            x2 = np.array(Image.open(self.folder_dir + self.f[self.train_or_validation_n][j,1])) / 255.  
            if np.random.rand() > self.flag:
                x1 = self.image_data_generator.random_transform(x1.astype('float32'))
            if np.random.rand() > self.flag:
                x2 = self.image_data_generator.random_transform(x2.astype('float32'))
            batch_x1[2*i] = x1
            batch_x2[2*i] = x2
            batch_y[2*i][1] = 1
            while True:
                index_1,index_2 = np.random.choice(self.path_list,2)
                if index_1[6] != index_2[6] and index_1[0:4] != index_2[0:4]:
                    break
            x1 = np.array(Image.open(self.folder_dir + index_1)) / 255.
            x2 = np.array(Image.open(self.folder_dir + index_2)) / 255.
            batch_x1[2*i+1] = x1
            batch_x2[2*i+1] = x2
            batch_y[2*i+1][0] = 1
            
        return [batch_x1,batch_x2], batch_y


class ImageDataGenerator_for_multiinput(pre_image.ImageDataGenerator):
            
    def flow(self, f, path_list, train_or_validation = 'train', flag = 0, batch_size=32, shuffle=True, seed=1217):
        
        return NumpyArrayIterator_for_Market1501(
            f, path_list, train_or_validation, flag, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
    
    def agumentation(self, X, rounds=1, seed=None):
        
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
        for r in range(rounds):
            for i in range(X.shape[0]):
                aX[i + r * X.shape[0]] = self.random_transform(X[i])
        X = aX
        return X

    
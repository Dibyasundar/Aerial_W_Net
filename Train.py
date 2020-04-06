import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.util.shape import view_as_windows as vaw
import scipy.io as io

from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, merge, Dropout, concatenate, LeakyReLU, Lambda, BatchNormalization
from keras.initializers import RandomNormal
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from keras.regularizers import l2
import keras as k
from keras.utils import to_categorical




if not os.path.exists('train.mat'):
    files=os.listdir('Data/training/input');
    k=0;
    windows=(512,512)
    steps=(500,500)
    for file in files:
        inp=plt.imread('Data/training/input/'+file)
        tar=plt.imread('Data/training/target/'+file[0:len(file)-1])
        inp=inp/255
        for i in range(inp.shape[2]):
            if i == 0:
                temp_patches=vaw(inp[:,:,0],windows,step=steps)
                sz=temp_patches.shape
                inp_patches=np.reshape(temp_patches,(sz[0]*sz[1],sz[2],sz[3],1))
            else:
                temp_patches=vaw(inp[:,:,i],windows,step=steps)
                sz=temp_patches.shape
                temp_patches=np.reshape(temp_patches,(sz[0]*sz[1],sz[2],sz[3],1))
                inp_patches=np.append(inp_patches,temp_patches,axis=3)
        tar_patches=vaw(tar,windows,step=steps)
        sz=tar_patches.shape
        tar_patches=np.reshape(tar_patches,(sz[0]*sz[1],sz[2],sz[3]))
        sz=tar_patches.shape
        for i in range(sz[0]):
            if (np.sum(tar_patches[i]/255)/(sz[1]*sz[2]))>=0.02 and np.mean(inp_patches[i])<=0.4:
                plt.imsave('Proc_train_data/inp_'+str(k)+'_0.bmp',inp_patches[i])
                t=tar_patches[i]
                plt.imsave('Proc_train_data/inp_'+str(k)+'_1.bmp',tar_patches[i]/255)
                k=k+1
    print('complete')
    files=os.listdir('Proc_train_data/');
    for i in range(int(len(files)/2)):
        img=plt.imread('Proc_train_data/inp_'+str(i)+'_0.bmp')
        tar=plt.imread('Proc_train_data/inp_'+str(i)+'_1.bmp')
        tar=np.mean(tar,axis=2)
        tar=(tar-np.min(tar))/(np.max(tar)-np.min(tar))
        img=np.expand_dims(img,axis=0)
        tar=np.expand_dims(tar,axis=0)
        if i==0:
            train_image=img;
            train_target=tar;
        else:
            train_image=np.append(train_image,img,axis=0);
            train_target=np.append(train_target,tar,axis=0);
    print(train_image.shape)
    print(train_target.shape)
    io.savemat('train.mat',mdict={'train_image':train_image,'train_target':train_target})
else:
    data=io.loadmat('train.mat')




train_image=data['train_image']
train_target=data['train_target']
train_target=np.expand_dims(train_target,axis=3)
cat_train_target=to_categorical(train_target)
print(train_image.shape)
print(train_target.shape)




def sum_mod(x):
    return (x[0]+x[1])/2
def w_model(inp_sz):
    inputs=Input(inp_sz, name='input')
    conv1=Conv2D(8,5,activation='relu',padding='same', name='conv1')(inputs)
    conv1=BatchNormalization(momentum=0.1)(conv1)
    pool1=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv1)
    conv2=Conv2D(16,3,activation='relu',padding='same', name='conv2')(pool1)
    conv2=BatchNormalization(momentum=0.1)(conv2)
    pool2=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2)
    conv3=Conv2D(32,3,activation='relu',padding='same', name='conv3')(pool2)
    conv3=BatchNormalization(momentum=0.1)(conv3)
    pool3=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv3)
    up1=concatenate([conv3,Conv2D(32,3,activation='relu',name='up1',padding='same')(UpSampling2D(size=(2,2))(pool3))],axis=3)
    up2=concatenate([conv2,Conv2D(16,3,activation='relu',name='up2',padding='same')(UpSampling2D(size=(2,2))(up1))],axis=3)
    up3=concatenate([conv1,Conv2D(8,3,activation='relu',name='up3',padding='same')(UpSampling2D(size=(2,2))(up2))],axis=3)
    out1=Conv2D(2,3,activation='softmax',name='out1',padding='same')(up3)
    conv01=Conv2D(4,11,activation='relu',padding='same', name='conv01')(out1)
    conv01=BatchNormalization(momentum=0.1)(conv01)
    pool01=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv01)
    conv02=Conv2D(8,3,activation='relu',padding='same', name='conv02')(pool01)
    conv02=BatchNormalization(momentum=0.1)(conv02)
    pool02=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv02)
    conv03=Conv2D(16,3,activation='relu',padding='same', name='conv03')(pool02)
    conv03=BatchNormalization(momentum=0.1)(conv03)
    pool03=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv03)
    up01=concatenate([conv03,Conv2D(16,3,activation='relu',name='up01',padding='same')(UpSampling2D(size=(2,2))(pool03))],axis=3)
    up02=concatenate([conv02,Conv2D(8,3,activation='relu',name='up02',padding='same')(UpSampling2D(size=(2,2))(up01))],axis=3)
    up03=concatenate([conv01,Conv2D(4,3,activation='relu',name='up03',padding='same')(UpSampling2D(size=(2,2))(up02))],axis=3)
    out2=Conv2D(2,3,activation='softmax',name='out2',padding='same')(up03)
    model=Model(input=inputs, output=[out2])
    return model



model=w_model((512,512,3))
model.summary()



model.save('model_base.h5')




model.compile(optimizer=k.optimizers.Adam(lr=0.001), loss={'out1':'categorical_crossentropy','out2':'categorical_crossentropy'}, metrics=['accuracy'] )


model.fit(train_image,{'out1':cat_train_target,'out2':cat_train_target},epochs=200,batch_size=10)
model.save_weights('model_weight.h5')


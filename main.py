# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on Jul 28, 2016

author: jakeret
forked for 3D BPConvNet
modified at Feb 2018
modified by : Kyong Jin (kyonghwan.jin@gmail.com)

# input data dimension : NCXYZ ( batch x channels x X x Y x Z )

running command example
python main.py --lr=1e-4 --output_path='logs/' --features_root=32 --layers=5 --restore=False
'''
import tensorflow as tf
from tf_unet import unet, util, image_util,layers
import h5py
import numpy as np
import os


flags=tf.app.flags
flags.DEFINE_integer('features_root',32,'learning rate')
flags.DEFINE_integer('layers',5,'number of depth')
flags.DEFINE_integer('channels',1,'number of channels in data')
flags.DEFINE_string('optimizer','adam','optimizing algorithm : adam / momentum')
flags.DEFINE_float('lr',1e-4,'learning rate')
flags.DEFINE_boolean('restore',True,'resotre model')
flags.DEFINE_string('output_path','logs/unet3d/wdec/','log folder')
flags.DEFINE_string('data_path','/Users/roservinals/Documents/EPFL/1B/DeepCNN/code/3D/3d/train/*.h5','log folder')
flags.DEFINE_string('test_path','/Users/roservinals/Documents/EPFL/1B/DeepCNN/code/3D/3d/test/*.h5','log folder')
flags.DEFINE_boolean('is_training',True,'training phase/ deploying phase')
conf=flags.FLAGS

if __name__ =='__main__':
    data_provider=image_util.ImageDataProvider_hdf5_vol(conf.data_path,conf.channels)
    data_provider_test=image_util.ImageDataProvider_hdf5_vol(conf.test_path,conf.channels,test=True)
    net=unet.Unet(layers=conf.layers,size_mask=[16,16,32],features_root=conf.features_root,channels=conf.channels, n_class=2,filter_size=3, cost='cross_entropy')

    if conf.is_training:
        #setup & trainig
        if conf.optimizer=='adam':
            trainer =unet.Trainer(net,batch_size=1,optimizer="adam",opt_kwargs=dict(beta1=0.9,learning_rate=conf.lr))
        elif conf.optimizer=='momentum':
            trainer =unet.Trainer(net,batch_size=1,optimizer="momentum",opt_kwargs=dict(momentum=0.99,learning_rate=conf.lr))

        path=trainer.train(data_provider,conf.output_path,training_iters=2,epochs=5,dropout=1,restore=True)

    else:
        save_path = os.path.join(conf.output_path, "model.cpkt")
        acc=0.0
        Ntest=364
        identifiers=list()
        x_t=list()
        y_t=list()
        pred=list()
        for i in np.arange(Ntest):
            print(repr(i)+"/"+repr(Ntest))
            x_test, y_test, name = data_provider_test(1)
            prediction = net.predict(save_path, x_test)
            #print(prediction)
            tmp_flag=np.where(prediction==np.amax(prediction),1,0)
            acc += np.float(np.sum(tmp_flag*y_test))
            
            x_t.append(x_test)
            y_t.append(y_test)
            pred.append(prediction)
            id=name.split("_")
            id_str=id[1]
            num_str=np.shape(id)
            for j in range(2,num_str[0]-2):
                id_str=id_str+'-'+id[j]
            identifiers.append(id_str)
        
        #print("Prediction avg class: ",(prediction))
        print("Testing Accurancy/patch: ",(acc/Ntest))

        acc_patient1=0
        scanned=list()
        for i in np.arange(Ntest):
            if i not in scanned:
                str_i=identifiers[i]
                indices = [j for j, x in enumerate(identifiers) if x == str_i]
                pre=np.zeros((1,2))
                for k in range(0,len(indices)):
                    idx_k=indices[k]
                    scanned.append(idx_k)
                    pred_k=pred[idx_k]
                    pre[0,0]=pred_k[0,0]+pre[0,0]
                    pre[0,1]=pred_k[0,1]+pre[0,1]
                    # if(len(indices)>1):
                    #     print('###########'+str(i)+'/'+str(idx_k)+'###########')
                    #     print('----------'+identifiers[idx_k]+'------------')
                    #     print(indices)
                    #     print(pred[idx_k])
                    #     print(y_t[idx_k])
                pre[0,0]=pre[0,0]/len(indices)
                pre[0,1]=pre[0,1]/len(indices)   
                tmp_flag=np.where(pre==np.amax(pre),1,0)
                acc_patient1 += np.float(np.sum(tmp_flag*y_t[i]))
        print("Testing Accurancy/patient1: ",(acc_patient1/len(scanned)))
        acc_patient2=0
        scanned=list()
        for i in np.arange(Ntest):
            if i not in scanned:
                str_i=identifiers[i]
                indices = [j for j, x in enumerate(identifiers) if x == str_i]
                pre=np.zeros((1,2))
                for k in range(0,len(indices)):
                    idx_k=indices[k]
                    scanned.append(idx_k)
                    pred_k=pred[idx_k]
                    pre[0,0]=pred_k[0,0]*pre[0,0]
                    pre[0,1]=pred_k[0,1]*pre[0,1]
                    # if(len(indices)>1):
                    #     print('###########'+str(i)+'/'+str(idx_k)+'###########')
                    #     print('----------'+identifiers[idx_k]+'------------')
                    #     print(indices)
                    #     print(pred[idx_k])
                    #     print(y_t[idx_k])
                tmp_flag=np.where(pre==np.amax(pre),1,0)
                acc_patient2 += np.float(np.sum(tmp_flag*y_t[i]))
        print("Testing Accurancy/patient2: ",(acc_patient2/len(scanned)))
        acc_patient3=0
        scanned=list()
        for i in np.arange(Ntest):
            if i not in scanned:
                str_i=identifiers[i]
                indices = [j for j, x in enumerate(identifiers) if x == str_i]
                pre=np.zeros((1,2))
                for k in range(0,len(indices)):
                    idx_k=indices[k]
                    scanned.append(idx_k)
                    pred_k=pred[idx_k]
                    pre[0,0]=max(pred_k[0,0],pre[0,0])
                    pre[0,1]=max(pred_k[0,1],pre[0,1])
                    # if(len(indices)>1):
                    #     print('###########'+str(i)+'/'+str(idx_k)+'###########')
                    #     print('----------'+identifiers[idx_k]+'------------')
                    #     print(indices)
                    #     print(pred[idx_k])
                    #     print(y_t[idx_k])
                tmp_flag=np.where(pre==np.amax(pre),1,0)
                acc_patient3 += np.float(np.sum(tmp_flag*y_t[i]))
        print("Testing Accurancy/patient3: ",(acc_patient3/len(scanned)))
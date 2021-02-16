#author: Zhiqin Xu 许志钦
#email: xuzhiqin@sjtu.edu.cn
#2019-09-24
# coding: utf-8

import sys
sys.path.insert(0,'../basicfolder')
import os,sys
import matplotlib
matplotlib.use('Agg')   
import pickle
import time  
import shutil 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt   
#from BasicFunc import mySaveFig, mkdir
from setup_mnist import MNIST
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import math

Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
pos=[Leftp,Bottomp,Widthp,Heightp]

def mkdir(fn):
    if not os.path.isdir(fn):
        os.mkdir(fn)
def mySaveFig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):
    if isax==1:
        #pltm.legend(fontsize=18)
        # plt.title(y_name,fontsize=14)
#        ax.set_xlabel('step',fontsize=18)
#        ax.set_ylabel('loss',fontsize=18)
        pltm.rc('xtick',labelsize=28)
        pltm.rc('ytick',labelsize=28)
        ax.set_position(pos, which='both')
    fnm='%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm='%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()
        

def get_clean_data(dat):
        return int(np.round(dat*1000))/1000
    
for arg in sys.argv[1:]:
    #print(np.type(arg))
    print(arg)
sBaseDir = 'scalestudym2/'
import platform
if platform.system()=='Windows':
    device_n="1"
    BaseDir = '../../../nn/%s'%(sBaseDir)
else:
    device_n="3"
    BaseDir = sBaseDir
    matplotlib.use('Agg')
    
n_image=45

ImgSize=28
data=  MNIST()
train_x_align=np.reshape(data.train_data[0:n_image,:,:,:],[n_image,ImgSize*ImgSize])
train_label=data.train_labels_one[0:n_image,:] 

    
neuron_num_set=[1000000]
#neuron_num_set=np.flip(neuron_num_set)
#neuron_num_set=[20000]
VarR={}
VarR['lr']=500
VarR['trysmalllr']=True
#VarR['s_gam1'] = [0.5,0.6,0.7,0.75,0.8,0.9,1,1.1,1.25,1.4,1.5,1.75]
VarR['s_gam1'] =np.flip(np.linspace(0.7,1.3,num=7))
VarR['isfail'] = False
VarR['s_gam1']=[1.3]
#VarR['s_gam1'] = [0.95,0.98,1.02,1.05]
#VarR['s_gam1'] = [1.2]
#VarR['s_gam1'] = [0.5,0.6,0.7,0.75,0.8,0.9,1]

gam2=0
gpu=0

lenarg=np.shape(sys.argv)[0] #Sys.argv[ ]其实就是一个列表，里边的项为用户输入的参数，关键就是要明白这参数是从程序外部输入的，而非代码本身的什么地方，要想看到它的效果就应该将程序保存了，从外部来运行程序并给出参数。
if lenarg>1:
    ilen=1
    while ilen<lenarg:
        if sys.argv[ilen]=='-g2':
            gam2=np.float32(sys.argv[ilen+1])  
        if sys.argv[ilen]=='-gpu':
            gpu=np.int32(sys.argv[ilen+1])  
        ilen=ilen+2
        
for gam1 in VarR['s_gam1']:
    #VarR['lr']=10
    for neu_num in neuron_num_set:
        VarR['lr']=0.0001
        print('neu num %s'%(neu_num))
        VarR['trysmalllr']=True
        VarR['isfail'] = False
        while VarR['trysmalllr']:
            tf.reset_default_graph()
            R={}
            ### used for saved all parameters and data
            R['hidden_units']=[neu_num]  #ONLY CONSIDER TWO-LAYER NETWORK
            
            
            R['output_dim']=1
            R['ActFuc']=0   ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate
            R['opti']='0'   ###  0: gd; 1: adam;  
            #if sys.argv[1:]:
            #    R['hidden_units']=[np.int32(sys.argv[1])]
            #    R['gpu']=sys.argv[2]
            #else:
            #    R['hidden_units']=[5000]
            #    R['gpu']=device_n
            R['VarR']=VarR  
            
            R['seed']=0
            R['tol']=1e-4
            R['Total_Step']=800000
            
            m=R['hidden_units'][0] 
            
                
            
            ### mkdir a folder to save all output
            R['input_dim']=np.shape(train_x_align)[1]
            R['train_size']=np.shape(train_x_align)[0]
            def get_y_func(xs):
                tmp=0
                for ii in range(R['input_dim']):
                    tmp+=xs[:,ii:ii+1]**2
                return tmp
            
            
            ### initialization standard deviation
            
            
            R['gam1']=gam1
            
            R['gam2']=gam2
            R['gpu']="%s"%(gpu)
            
            R['beta1exp']=0
            
            R['beta2exp']=R['beta1exp']-R['gam2']
            R['alphaexp']=R['gam1']- R['beta1exp']- R['beta2exp']
            
            
            R['alpha']=1/m**R['alphaexp']
            R['beta1']=1/m**R['beta1exp'] # for weight
            R['beta2']=1/m**R['beta2exp']# for bias terms2
            R['kappa']=R['alpha']*R['beta1']*R['beta2']
            
            R['learning_rate']=VarR['lr']/R['alpha']*(R['beta1']*R['beta2'])**0.5
             
            os.environ["CUDA_VISIBLE_DEVICES"]='%s'%(R['gpu']) 
            
        #    R['learning_rate']=10000
            mkdir(BaseDir)
            
            if R['gam2']<0:
                sds='neg'
            else:
                sds='pos'
            
            BaseDir_a='%s/gamma%s%.3f/'%(BaseDir,sds,abs(R['gam2']))
            mkdir(BaseDir_a)
            BaseDir_b='%s/sf%.3ff%.3f/'%(BaseDir_a,R['beta1exp'],R['beta2exp'])
            mkdir(BaseDir_b)
            BaseDir2='%s/%.3ff%.3ff%.3f/'%(BaseDir_b,R['alphaexp'],R['beta1exp'],R['beta2exp'])
            mkdir(BaseDir2)
            subFolderName = '%sr%s'%(R['hidden_units'][0],np.random.randint(0,high=10000)) 
            FolderName = '%s%s/'%(BaseDir2,subFolderName)
            mkdir(FolderName)
            R['FolderName'] = FolderName
            
            shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
            if  not platform.system()=='Windows':
                shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))
                
            
            
            R['ASI']=0
            
            R['learning_rateDecay']=0
            
            ### setup for activation function
            
            
            plotepoch=100
            R['batch_size']=45 # int(np.floor(R['train_size'])) ### batch size
            
            R['issave']=False
             ### the training step. Set a big number, if it converges, can manually stop training 
            
            R['FolderName']=FolderName   ### folder for save images
            
            print(R)  
            R['train_inputs']=train_x_align
            train_inputs=R['train_inputs'] 
            R['y_true_train']=train_label 
            
            t0=time.time() 
            
            def add_layer2(x,input_dim = 1,output_dim = 1,isresnet=0,astddev=0.05,
                           bstddev=0.05,ActFuc=0,seed=0, name_scope='hidden'):
                if not seed==0:
                    tf.set_random_seed(seed)
                
                with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
                    ua_w = tf.get_variable(name='ua_w', initializer=astddev)
                    ua_b = tf.get_variable(name='ua_b', initializer=bstddev) 
                    z=tf.matmul(x, ua_w) + ua_b
                    
                    
                    if ActFuc==1:
                        output_z = tf.nn.tanh(z)
                        print('tanh')
                    elif ActFuc==3:
                        output_z = tf.sin(z)
                        print('sin')
                    elif ActFuc==0:
                        output_z = tf.nn.relu(z)
                        print('relu')
                    elif ActFuc==4:
                        output_z = z**50
                        print('z**50')
                    elif ActFuc==5:
                        output_z = tf.nn.sigmoid(z)
                        print('sigmoid')
                        
                    L2Wight= tf.nn.l2_loss(ua_w) 
                    if isresnet and input_dim==output_dim:
                        output_z=output_z+x
                    return output_z,ua_w,ua_b,L2Wight
            
            def getWini(hidden_units=[10,20,40],input_dim = 1,output_dim_final = 1):
                
                hidden_num = len(hidden_units)
                #print(hidden_num)
                add_hidden = [input_dim] + hidden_units;
                
                w_Univ0=[]
                b_Univ0=[]
                
                for i in range(hidden_num):
                    input_dim = add_hidden[i]
                    output_dim = add_hidden[i+1]
                    ua_w=np.float32(np.random.normal(loc=0.0,scale=R['beta2'],size=[input_dim,output_dim]))
                    ua_b=np.float32(np.random.normal(loc=0.0,scale=R['beta2'],size=[output_dim]))
                    w_Univ0.append(ua_w)
                    b_Univ0.append(ua_b)
                ua_w=np.float32(np.random.normal(loc=0.0,scale=R['beta1'],size=[hidden_units[hidden_num-1], output_dim_final]))
                #ua_b=np.float32(np.random.normal(loc=0.0,scale=R['beta1'],size=[output_dim_final]))
                w_Univ0.append(ua_w)
                #b_Univ0.append(ua_b)
                return w_Univ0, b_Univ0
            
            
            
            def univAprox2(x0, hidden_units=[100],input_dim = 1,output_dim_final = 1,
                           isresnet=0, ActFuc=0,seed=0,ASI=1):
                if seed==0:
                    seed=time.time()
                # The simple case is f: R -> R 
                hidden_num = len(hidden_units)
                #print(hidden_num)
                add_hidden = [input_dim] + hidden_units;
                
                w_Univ=[]
                b_Univ=[] 
                L2w_all=0
                
                w_Univ0, b_Univ0=getWini(hidden_units=hidden_units,input_dim = input_dim,output_dim_final = output_dim_final)
                
            
                output=x0
                
                
                for i in range(hidden_num):
                    input_dim = add_hidden[i]
                    output_dim = add_hidden[i+1]
                    print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
                    name_scope = 'hidden' + np.str(i+1)
                        
                    output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                                           astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                                           seed=seed, name_scope= name_scope)
                    w_Univ.append(ua_w)
                    b_Univ.append(ua_b)
                    L2w_all=L2w_all+L2Wight_tmp
                
                ua_we = tf.get_variable(
                        name='ua_we'
                        #, shape=[hidden_units[hidden_num-1], output_dim_final]
                        , initializer=w_Univ0[-1]
                    )
            
                
                z1 = tf.matmul(output, ua_we)*R['alpha']
                w_Univ.append(ua_we)
                #b_Univ.append(ua_be)
                
                # you can ignore this trick for now. Consider ASI=False
                if ASI:
                    output=x0
                    for i in range(hidden_num):
                        input_dim = add_hidden[i]
                        output_dim = add_hidden[i+1]
                        print('input_dim:%s, output_dim:%s'%(input_dim,output_dim))
                        name_scope = 'hidden' + np.str(i+1+hidden_num)
                        output,ua_w,ua_b,L2Wight_tmp=add_layer2(output,input_dim,output_dim,isresnet=isresnet,
                                                           astddev=w_Univ0[i],bstddev=b_Univ0[i], ActFuc=ActFuc,
                                                           seed=seed, name_scope= name_scope)
                    ua_we = tf.get_variable(
                            name='ua_wei2'
                            #, shape=[hidden_units[hidden_num-1], output_dim_final]
                            , initializer=-w_Univ0[-1]
                        )
            
                    z2 = tf.matmul(output, ua_we)*R['alpha']
                else:
                    z2=0
                z=(z1+z2)
                return z,w_Univ,b_Univ,L2w_all
            
            #with tf.device('/gpu:%s'%(R['gpu'])):
            with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
                # Our inputs will be a batch of values taken by our functions
                x = tf.placeholder(tf.float32, shape=[None, R['input_dim']], name="x")
                y_true = tf.placeholder_with_default(input=[[0.0]], shape=[None, R['output_dim']], name="y")
                in_learning_rate= tf.placeholder_with_default(input=1e-3,shape=[],name='lr')
                y,w_Univ,b_Univ,_ = univAprox2(x, R['hidden_units'],input_dim = R['input_dim'],
                                                        ActFuc=R['ActFuc'],
                                                        seed=R['seed'],ASI=R['ASI'])
                
                loss=tf.reduce_mean(tf.square(y_true-y))
                # We define our train operation using the Adam optimizer
                #adam = tf.compat.v1.train.AdamOptimizer(learning_rate=in_learning_rate)
                if R['opti']=='0':
                    gd=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=in_learning_rate)
                elif R['opti']=='1':
                    gd=tf.compat.v1.train.AdamOptimizer(learning_rate=in_learning_rate)
                
                train_op = gd.minimize(loss)
            
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True  
            sess = tf.Session(config=config)
            
            sess.run(tf.global_variables_initializer())
            if R['issave']:
                saver = tf.train.Saver()  
                    
            class model():
                def __init__(self): 
                    R['y_train']=[]
#                    R['y_test']=[]
#                    R['loss_test']=[]
                    R['loss_train']=[]
                    
                    R['theta_diff']=[]
                    
                    R['amp_av']=[]
                    R['angle_v']=[]
                    
                    R['theta_diff_all']=[]
                    R['rela_theta_diff_all']=[]
                    
                    R['theta_diff_rescale_all']=[]
                    R['rela_theta_diff_rescale_all']=[]
                    
                    R['theta0']=0 
                    sess.run(tf.global_variables_initializer())
                    if R['issave']:
                        nametmp='%smodel/'%(FolderName)
                        mkdir(nametmp)
                        saver.save(sess, "%smodel.ckpt"%(nametmp))
                    
#                    y_test, loss_test_tmp,w_Univ_tmp,b_Univ_tmp= sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: test_inputs, y_true: R['y_true_test']})
                    
                    loss_train_tmp,y_train,w_Univ_tmp,b_Univ_tmp = sess.run([loss,y,w_Univ,b_Univ], feed_dict={x: train_inputs, y_true: R['y_true_train']})
                    theta, amp_av,angle_v,theta_rescale=self.wb2theta(w_Univ_tmp,b_Univ_tmp)
                    R['amp_av0']=np.squeeze(amp_av)
                    R['angle_v0']=np.squeeze(angle_v)
                    R['theta0']=theta
                    theta0_norm=np.sqrt(np.mean(np.square((theta.ravel()))))
                    R['theta0_norm']=theta0_norm
                    
                    R['theta0_rescale']=theta_rescale
                    theta0_norm_rescale=np.sqrt(np.mean(np.square((theta_rescale.ravel()))))
                    R['theta0_norm_rescale']=theta0_norm_rescale
                    
                    #y_train,loss_train_tmp,w_Univ_tmp,b_Univ_tmp= sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: train_inputs, y_true: R['y_true_train']})
                    #R['y_train']=y_train
#                    R['y_test']=y_test
#                    R['loss_test'].append(loss_test_tmp)
                    R['loss_train'].append(loss_train_tmp)
                    
                    R['amp_av_end']=np.squeeze(amp_av)
                    R['angle_v_end']=np.squeeze(angle_v)
                    
                    self.ploty(name='ini') 
            
                    self.plotQv(name='ini')
                def run_onestep(self):
                    
                    #y_train,loss_train_tmp,w_Univ_tmp,b_Univ_tmp= sess.run([y,loss,w_Univ,b_Univ],feed_dict={x: train_inputs, y_true: R['y_true_train']})
                        
                    if R['train_size']>R['batch_size']:
                        indperm = np.random.permutation(R['train_size'])
                        nrun_epoch=np.int32(R['train_size']/R['batch_size'])
                        
                        for ijn in range(nrun_epoch):
                            ind = indperm[ijn*R['batch_size']:(ijn+1)*R['batch_size']] 
                            _= sess.run(train_op, feed_dict={x: train_inputs[ind], y_true: R['y_true_train'][ind],
                                                              in_learning_rate:R['learning_rate']})
                    else:
                        _,loss_train_tmp,y_train,w_Univ_tmp,b_Univ_tmp = sess.run([train_op,loss,y,w_Univ,b_Univ], feed_dict={x: train_inputs, y_true: R['y_true_train'],
                                                              in_learning_rate:R['learning_rate']})
                    #R['learning_rate']=R['learning_rate']*(1-R['learning_rateDecay'])
                    
                    loss_train_tmp,w_Univ_tmp,b_Univ_tmp = sess.run([loss,w_Univ,b_Univ], feed_dict={x: train_inputs, y_true: R['y_true_train']})
                    #R['y_train']=y_train
                    R['loss_train'].append(loss_train_tmp)
                def run(self,step_n=1):
                    if R['issave']:
                        nametmp='%smodel/model.ckpt'%(FolderName)
                        saver.restore(sess, nametmp)
                    for ii in range(step_n):
                        self.run_onestep()
                        if R['loss_train'][-1]>R['loss_train'][-2]:
                            VarR['lr']=VarR['lr']/2
                            shutil.rmtree(FolderName)
                            VarR['trysmalllr']=True
                            print('lr is too large, now try %s'%(VarR['lr']))
                            return
                        if R['loss_train'][-1]<R['tol']: 
                            VarR['trysmalllr']=False
                            self.runtest(ii)
                            self.plotloss()
                            self.ploty(name='%s'%(ii))
                            self.plotQv(name='%s'%(ii))
                            self.savefile()
                            if R['issave']:
                                nametmp='%smodel/'%(FolderName)
                                shutil.rmtree(nametmp)
                                saver.save(sess, "%smodel.ckpt"%(nametmp))
                            break
                        
                        if ii>step_n-2:
                            VarR['isfail']=True 
                            shutil.rmtree(FolderName)
                            VarR['trysmalllr']=False
                            print('cannot find a lr to fast converge')
                            return
                        
#                        if ii==0:
#                            print('initial %s'%(np.max(R['y_train'])))
                            
                        if ii%plotepoch==0:
                            self.runtest(ii)
                            
                            self.plotloss()
                            self.ploty(name='%s'%(ii))
                            self.plotQv(name='%s'%(ii))
                            self.savefile()
                            if R['issave']:
                                
                                nametmp='%smodel/'%(FolderName)
                                shutil.rmtree(nametmp)
                                saver.save(sess, "%smodel.ckpt"%(nametmp))
                        
                            
                def runtest(self,ii):
                    w_Univ_tmp,b_Univ_tmp= sess.run([w_Univ,b_Univ],feed_dict={x: train_inputs, y_true: R['y_true_train']})
                    theta, amp_av,angle_v,theta_rescale=self.wb2theta(w_Univ_tmp,b_Univ_tmp)
                    R['theta_end']=theta
                    R['theta_end_rescale']=theta_rescale
                    R['amp_av_end']=np.squeeze(amp_av)
                    R['angle_v_end']=np.squeeze(angle_v)
                    theta_diff=R['theta_end']-R['theta0']
                    diff_norm=np.sqrt(np.mean(np.square((theta_diff.ravel()))))
                    rela_diff_norm=diff_norm/R['theta0_norm']
                    R['theta_diff_all'].append(diff_norm)
                    R['rela_theta_diff_all'].append(rela_diff_norm)
                    
                    theta_diff_rescale=R['theta_end_rescale']-R['theta0_rescale']
                    diff_norm_rescale=np.sqrt(np.mean(np.square((theta_diff_rescale.ravel()))))
                    rela_diff_norm_rescale=diff_norm_rescale/R['theta0_norm_rescale']
                    R['theta_diff_rescale_all'].append(diff_norm_rescale)
                    R['rela_theta_diff_rescale_all'].append(rela_diff_norm_rescale)
                    
                    
#                    R['loss_test'].append(loss_test_tmp)
                    print('time elapse: %.3f'%(time.time()-t0))
                    print('model, epoch: %d, train loss: %f' % (ii,R['loss_train'][-1]))
                def L2norm(self,var,axis=0):
                    var2=np.square(var)
                    var_norm=np.sqrt(np.sum(var2,axis=axis,keepdims=True))
                    return var_norm
                
                def wb2theta(self,w_Univ_tmp,b_Univ_tmp):
                    v=np.concatenate((w_Univ_tmp[0],b_Univ_tmp),axis=0)
                    a=np.transpose(w_Univ_tmp[1])
                    theta=np.concatenate((v,a),axis=0)
                    theta_rescale=np.concatenate((1/R['beta2']*v,1/R['beta1']*a),axis=0)
                    amp_v=self.L2norm(v,0)
                    amp_av=a*amp_v
                    dim=np.shape(w_Univ_tmp[0])[0]+1
                    angle_v=np.sum(v,axis=0)/amp_v/np.sqrt(dim)
                    return theta, amp_av,angle_v,theta_rescale
                
                def plotloss(self):
                    plt.figure()
                    ax = plt.gca()
#                    y1 = R['loss_test']
                    y2 = R['loss_train']
                    #plt.plot(y1,'ro',label='Test')
                    plt.plot(y2,'g*',label='Train')
                    ax.set_xscale('log')
                    ax.set_yscale('log')                
                    plt.legend(fontsize=18)
                    plt.title('loss',fontsize=15)
                    fntmp = '%sloss'%(FolderName)
                    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                    
                        
                def ploty(self,name=''):
                    plt.figure()
                    ax = plt.gca()
                    plt.plot(R['angle_v_end'],R['amp_av_end'],'k.',markersize=14,label='angle')
                    #plt.title('g2u',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%sangle%s'%(FolderName,name)
                    mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                    if R['input_dim']==2:
                        # Make data.
                        X = np.arange(R['x_start'], R['x_end'], 0.1)
                        Y = np.arange(R['x_start'], R['x_end'], 0.1)
                        X, Y = np.meshgrid(X, Y)
                        xy=np.concatenate((np.reshape(X,[-1,1]),np.reshape(Y,[-1,1])),axis=1)
                        Z = np.reshape(get_y_func(xy),[len(X),-1])
                        fp = plt.figure()
                        ax = fp.gca(projection='3d')
                        # Plot the surface.
                        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                        # Customize the z axis.
                        #ax.set_zlim(-2.01, 2.01)
                        ax.zaxis.set_major_locator(LinearLocator(5))
                        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                        # Add a color bar which maps values to colors.
                        fp.colorbar(surf, shrink=0.5, aspect=5)
                        ax.scatter(train_inputs[:,0], train_inputs[:,1], R['y_train'])
                        fntmp = '%s2du%s'%(FolderName,name)
                        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                    if R['input_dim']==1:
                        R['y_test'] = sess.run(y,feed_dict={x: test_inputs})
                        plt.figure()
                        ax = plt.gca()
                        y1 = R['y_test'] 
                        y3 = R['y_true_train']
                        plt.plot(test_inputs,y1,'r-',label='Test')
                        #plt.plot(train_inputs,y2,'go',label='Train')
                        plt.plot(train_inputs,y3,'b*',label='True')
                        #plt.title('g2u',fontsize=15)        
                        plt.legend(fontsize=18) 
                        fntmp = '%su_m%s'%(FolderName,name)
                        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
                        
                def savefile(self):
                    with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(R, f, protocol=4)
                     
                    text_file = open("%s/Output.txt"%(FolderName), "w")
                    for para in R:
                        if np.size(R[para])>20:
                            continue
                        text_file.write('%s: %s\n'%(para,R[para]))
                    for para in sys.argv: 
                        text_file.write('%s  '%(para))
                    text_file.close()
                    
                def plotQv(self,name=''): 
                    fp=plt.figure()
                    ax = plt.gca()
                    plt.plot(R['theta_diff_all'],'k.',markersize=14,label='angle')
                    #plt.title('g2u',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%stheta_diff_all'%(FolderName)
                    mySaveFig(plt,fntmp,ax=ax,fp=fp,isax=1,iseps=0)
                    
                    fp=plt.figure()
                    ax = plt.gca()
                    plt.plot(R['rela_theta_diff_all'],'k.',markersize=14,label='angle')
                    #plt.title('g2u',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%srela_theta_diff_all'%(FolderName)
                    mySaveFig(plt,fntmp,ax=ax,fp=fp,isax=1,iseps=0)
                    
                    fp=plt.figure()
                    ax = plt.gca()
                    plt.plot(R['theta_diff_rescale_all'],'k.',markersize=14,label='angle')
                    #plt.title('g2u',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%stheta_diff_rescale_all'%(FolderName)
                    mySaveFig(plt,fntmp,ax=ax,fp=fp,isax=1,iseps=0)
                    
                    fp=plt.figure()
                    ax = plt.gca()
                    plt.plot(R['rela_theta_diff_rescale_all'],'k.',markersize=14,label='angle')
                    #plt.title('g2u',fontsize=15)        
                    plt.legend(fontsize=18) 
                    fntmp = '%srela_theta_diff_rescale_all'%(FolderName)
                    mySaveFig(plt,fntmp,ax=ax,fp=fp,isax=1,iseps=0)
                    
                    
                            
                        
            model1=model()
            model1.run(R['Total_Step'])
            if VarR['trysmalllr']:
                continue
        #    BaseDir2='C:/Users/user/Desktop/rescale/0.25f0f0/'
#            BaseDir2='D:/researchlocala/rescale/systemexp/gammap0/sf0f0/1.750f0f0/'
#            BaseDir2='D:/researchlocala/rescale/systemexp/gammap0/sf0f0/0.500f0f0/'
            FolderName=BaseDir2
            m_all=[]
            angle_v0_all=[]
            amp_av0_all=[]
            angle_v_end_all=[]
            amp_av_end_all=[]
            theta_diff_all=[]
            rela_theta_diff_all=[]
            theta_diff_rescale_all=[]
            rela_theta_diff_rescale_all=[]
            target_folder=BaseDir2
            all_sub=os.listdir(target_folder) 
            a_diff_rescale_all=[]
            w_diff_rescale_all=[]
            for ii, sub in enumerate(all_sub):
                fd=target_folder+'/'+sub
                if os.path.isdir(fd):
                    for fname in os.listdir(fd):
                        if not fname[-4:] == '.pkl':
                            continue
                        with open(target_folder+'/'+sub+'/'+fname,'rb') as f:
                            R = pickle.load(f)
                        break
                    print(sub)
                    m_all.append(R['hidden_units'][0])
                    angle_v0_all.append(np.squeeze(R['angle_v0']))
                    amp_av0_all.append(np.squeeze(R['amp_av0']))
                    angle_v_end_all.append(np.squeeze(R['angle_v_end']))
                    amp_av_end_all.append(np.squeeze(R['amp_av_end']))
                    theta_diff_all.append(R['theta_diff_all'][-1])
                    rela_theta_diff_all.append(R['rela_theta_diff_all'][-1])
                    theta_diff_rescale_all.append(R['theta_diff_rescale_all'][-1])
                    rela_theta_diff_rescale_all.append(R['rela_theta_diff_rescale_all'][-1])
                    
                   
                    
                    theta_diff_rescale=R['theta_end_rescale']-R['theta0_rescale'] 
                    #### I USE RELATIVE NOW, BUT DONOT change the notation!!!!  They are quite close
                    diff_a_rescale=np.sqrt(np.mean(np.square((theta_diff_rescale[2,:].ravel()))))/np.sqrt(np.mean(np.square((R['theta0_rescale'][2,:].ravel()))))
                    diff_w_rescale=np.sqrt(np.mean(np.square((theta_diff_rescale[0:2,:].ravel()))))/np.sqrt(np.mean(np.square((R['theta0_rescale'][0:2,:].ravel()))))
                    a_diff_rescale_all.append(diff_a_rescale) 
                    w_diff_rescale_all.append(diff_w_rescale) 
                    
                
                
                    fp=plt.figure()
                    ax=plt.gca()
                    #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
                    plt.plot(angle_v_end_all[-1],amp_av_end_all[-1],'r.',markersize=10,label=r'final')
                    plt.plot(angle_v0_all[-1],amp_av0_all[-1],'c.',markersize=10,label=r'initial')
                    #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
                    ftsz=28
                    plt.xlabel('angle',fontsize=ftsz)
                    plt.ylabel('amp',fontsize=ftsz)
                    plt.rc('xtick',labelsize=ftsz)
                    plt.rc('ytick',labelsize=ftsz)
                    plt.yticks(fontsize=ftsz) 
                    #plt.legend(fontsize=18,ncol=2)
                    #plt.ylim([-1.25,-0.15])
                    ax.set_position(pos, which='both')
                    fntmp = '%sangleamp%s'%(FolderName,m_all[-1]) 
                    mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
        #    
            coe=np.polyfit(np.log(m_all),np.log(theta_diff_all),1)    
            fp=plt.figure()
            ax=plt.gca()
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(m_all,theta_diff_all,'b.',linewidth=4,markersize=14,label='slope=%.3f'%(coe[0])) 
            #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
            ftsz=28
            plt.xlabel('m',fontsize=ftsz)
            plt.title(r'$||\theta^{*}-\theta_{0}||_{2}/\sqrt{m}$',fontsize=ftsz)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.legend(fontsize=18)
            #plt.ylim([1e-1,1e0])
        #    plt.ylim([1e-6,1e-4])
        #    plt.xlim([8e2,1e5])
        #    plt.xticks([1e3,1e4,1e5])
            ax.set_position(pos, which='both')
            fntmp = '%sthetadiff'%(FolderName) 
            mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
            
            coe=np.polyfit(np.log(m_all),np.log(rela_theta_diff_all),1)    
            fp=plt.figure()
            ax=plt.gca()
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(m_all,rela_theta_diff_all,'b.',linewidth=4,markersize=14,label='slope=%.3f'%(coe[0])) 
            #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
            ftsz=28
            plt.xlabel('m',fontsize=ftsz)
            plt.title(r'$||\theta^{*}-\theta_{0}||_{2}/||\theta_{0}||_{2}$',fontsize=ftsz)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            ax.set_yscale('log')
            ax.set_xscale('log')
            #plt.ylim([1e-1,1e0])
            plt.legend(fontsize=18,ncol=2)
        #    plt.ylim([1e-6,1e-4])
        #    plt.xlim([8e2,1e5])
        #    plt.xticks([1e3,1e4,1e5])
            #plt.legend(fontsize=18,ncol=2)
            #plt.ylim([-1.25,-0.15])
            ax.set_position(pos, which='both')
            fntmp = '%srelathetadiff'%(FolderName) 
            mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
            
            coe=np.polyfit(np.log(m_all),np.log(theta_diff_rescale_all),1)    
            fp=plt.figure()
            ax=plt.gca()
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(m_all,theta_diff_rescale_all,'b.',linewidth=4,markersize=14,label='slope=%.3f'%(coe[0])) 
            #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
            ftsz=28
            plt.xlabel('m',fontsize=ftsz)
            plt.title(r'$||\bar{\theta}^{*}-\bar{\theta}_{0}||_{2}/\sqrt{m}$',fontsize=ftsz)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.legend(fontsize=18,ncol=2)
            #plt.ylim([1e-1,1e0])
        #    plt.ylim([1e-6,1e-4])
        #    plt.xlim([8e2,1e5])
        #    plt.xticks([1e3,1e4,1e5])
            ax.set_position(pos, which='both')
            fntmp = '%sthetadiff_rescale'%(FolderName) 
            mySaveFig(plt,fntmp,ax=ax,isax=1,fp=fp,iseps=0)
            
            coe=np.polyfit(np.log(m_all),np.log(rela_theta_diff_rescale_all),1)   
            fp=plt.figure()
            ax=plt.gca()
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(m_all,rela_theta_diff_rescale_all,'b.',linewidth=4,markersize=14,label='slope=%.3f'%(coe[0])) 
            #plt.plot(x_tst,y_ini,'c-',linewidth=2,label='initial-nn')
            ftsz=28
            plt.xlabel('m',fontsize=ftsz)
            plt.title(r'RD($\theta^{*}$)',fontsize=ftsz)
            #plt.title(r'$||\bar{\theta}^{*}-\bar{\theta}_{0}||_{2}/||\bar{\theta}_{0}||_{2}$',fontsize=ftsz)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.ylim([9e-2,1.2e0])
#            plt.legend(fontsize=ftsz,ncol=1)
        #    plt.ylim([1e-6,1e-4])
        #    plt.xlim([8e2,1e5])
            plt.yticks([1e-1,1])
            #plt.legend(fontsize=18,ncol=2)
            #plt.ylim([-1.25,-0.15])
            ax.set_position(pos, which='both')
            fntmp = '%srelathetadiff_rescale'%(FolderName) 
            mySaveFig(plt,fntmp,fp=fp,iseps=0)



            coe=np.polyfit(np.log(m_all),np.log(w_diff_rescale_all),1)   
            fp=plt.figure()
            ax=plt.gca()
            #plt.plot(x_tgt,y_tgt,'k-',linewidth=2,label='auxiliary')
            plt.plot(m_all,w_diff_rescale_all,'b.',linewidth=4,markersize=14,label='data') 
            plt.plot(m_all,np.exp(np.log(m_all)*coe[0]+coe[1]),'grey',label='slope=%.3f'%(coe[0]))
            ftsz=22
            plt.xlabel('m',fontsize=ftsz)
            plt.title(r'RD($\theta^{*}_{w}$)',fontsize=ftsz)
            #plt.title(r'$||\bar{\theta}^{*}-\bar{\theta}_{0}||_{2}/||\bar{\theta}_{0}||_{2}$',fontsize=ftsz)
            plt.rc('xtick',labelsize=ftsz)
            plt.rc('ytick',labelsize=ftsz)
            plt.yticks(fontsize=ftsz) 
            ax.set_yscale('log')
            ax.set_xscale('log')
            plt.ylim([6,1e2])
            plt.legend(fontsize=20,ncol=1)
        #    plt.ylim([1e-6,1e-4])
        #    plt.xlim([8e2,1e5])
            plt.yticks([10,1e2])
            #plt.legend(fontsize=18,ncol=2)
            #plt.ylim([-1.25,-0.15])
            ax.set_position(pos, which='both')
            fntmp = '%swreladiffrescale'%(FolderName) 
            mySaveFig(plt,fntmp,fp=fp,iseps=0)


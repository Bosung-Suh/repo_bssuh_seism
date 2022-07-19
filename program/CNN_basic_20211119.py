# -*- coding: utf-8 -*-
#IMPORT


#000000INPUT
    #Width*Height*channel


#111111Convolution layer
    #INPUT=Width*Height*channel

    #API=tf.keras.conv2D
"""  __init__(      #class
     filters,        #output filter channel number
     kernel_size,        #conv layer filter size(integer 3, tuple, list)
     strides=(1,1),      #filter movement at one time
     padding='valid',        #valid(p=0)/same(stride=1을 기준으로 input=output)
     data_format=None,       #channels_last(default)=(batch,height,width,channels)
     dilation_rate=(1,1),        
     activation=None,        #함수 활성화(relu, linear...)
     use_bias=True,
     kernel_initializer='glorot_uniform',    #kernel dimension: {height, width, in_channel, out_channel}
     bias_initializer='zeros',
     kernel_regularizer=None,
     bias_regularizer=None,
     activity_regularizer=None,
     kernel_constraint=None,
     bias_constraint=None,
     **kwargs
 ) """
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
tf.enable_eager_execution()

    #Toy image(3*3)=input
image = tf.constant([[[[1],[2],[3]],
                   [[4],[5],[6]], 
                   [[7],[8],[9]]]], dtype=np.float32)       #(1*3*3*1) batch*height*width* channel
##print(image.shape)
##plt.imshow(image.numpy().reshape(3,3), cmap='Greys')
##plt.show()

    #filter(weight)=Width*Height*channel
weight = np.array([[[[1.]],[[1.]]],
                   [[[1.]],[[1.]]]])        #(2*2*1*1)
##print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=1, kernel_size=2, padding='VALID', 
                             kernel_initializer=weight_init)(image)     #padding 넣을 거면 padding='SAME'
##print("conv2d.shape", conv2d.shape)
##print(conv2d.numpy().reshape(2,2))
##plt.imshow(conv2d.numpy().reshape(2,2), cmap='gray')
##plt.show()

    #3 filters
weight = np.array([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                   [[[1.,10.,-1.]],[[1.,10.,-1.]]]])        #2*2*1(channel)*3, (1 1 1 1), (10 10 10 10), (-1 -1 -1 -1) 이렇게 3개의 2*2 filter
##print("weight.shape", weight.shape)
weight_init = tf.constant_initializer(weight)
conv2d = keras.layers.Conv2D(filters=3, kernel_size=2, padding='SAME',
                             kernel_initializer=weight_init)(image)     #filters=____
##print("conv2d.shape", conv2d.shape)
feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    print(feature_map.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(feature_map.reshape(3,3), cmap='gray')
plt.show()


#Relu: convolution 결과값 중에서 음수는 0으로, 양수는 그대로 통과
    #Load mnist
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist # fasion_mnist, cifar10, cifar100
tf.enable_eager_execution()
def relu() :
    return tf.keras.layers.Activation(tf.keras.activations.relu)


#222222Pooling
    #input(input의 channel 수=output의 channel 수)
image = tf.constant([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
    #Maxpooling
        #API=tf.keras.layers.Maxpool2D
"""  __init(
     pool_size=(2,2),    #pooling filter size
     strides=None,     #filter move at one time
     padding='valid',
     data_format=None,
     **kwargs
 ) """
pool = keras.layers.MaxPool2D(pool_size=(2,2), strides=1, padding='VALID')(image)       #padding 넣을 거면 padding='SAME'
print(pool.shape)
print(pool.numpy())

#111111+222222(MNIST data)
    #data load
mnist = keras.datasets.mnist    #input=mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']    #분류할 class 이름
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    #test data, train data 분리
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.    #train/test image를 255로 나누어서 [0,1]로 scaling
#img = train_images[0]    #0번 이미지 불러와서 미리보기
#plt.imshow(img, cmap='gray')
#plt.show()
    #data processing
img = img.reshape(-1,28,28,1)    #image to 4D(-1=batch 값 알아서(1 image))
img = tf.convert_to_tensor(img)    #numpy ndarray 데이터를 API에 집어넣기 위해 tensor로 변환
    #conv2d
weight_init = keras.initializers.RandomNormal(stddev=0.01)
conv2d = keras.layers.Conv2D(filters=5, kernel_size=3, strides=(2, 2), padding='SAME', 
                             kernel_initializer=weight_init)(img)    #28*28 to 14*14    #conv2d의 입력으로 processing한 dataset을 불러옴
#print(conv2d.shape)    #(1, 14, 14, 5)
""" feature_maps = np.swapaxes(conv2d, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1,5,i+1), plt.imshow(feature_map.reshape(14,14), cmap='gray')
plt.show() """    #각각의 filter 연산 결과(feature map) plot
    #pooling
pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(conv2d)    #14*14 to 7*7    #pooling의 입력으로 conv2d의 결과을 불러옴
#print(pool.shape)    #(1, 7, 7, 5)

""" feature_maps = np.swapaxes(pool, 0, 3)
for i, feature_map in enumerate(feature_maps):
    plt.subplot(1,5,i+1), plt.imshow(feature_map.reshape(7, 7), cmap='gray')
plt.show() """    #각각의 feature map에 대한 pooling 결과 plot

#333333Fully connected neutral network
# conv2d+pooling 결과로 나오는 입체 데이터를 일렬로 된 벡터 형태로 펴서 전부 다 연결, 연결한 출력값을 softmax에 통과


#000000labeling

#CNN의 전체 과정(MNIST Data를 예시로)
#28*28image -> 3*3conv & 2*2maxpool -> 3*3conv & 2*2maxpool -> 3*3conv & 2*2maxpool -> fully connected layer 2단(256 > 10)
""" 
대략적인 과정
1. set hyper parameters(learning rate, training epochs, batch size, etc)
2. make data pipelining(use tf.data)    #dataset을 로드하고 설정한 batch size 만큼 데이터를 가져와서 network에 전달
3. build neural network model(use tf.keras.sequential API)
4. define loss function(cross entropy)    #classification 문제이므로 cross entropy
5. calculate gradient(use tf.GradientTape)    #weight에 대한 gradient 계산
6. select optimizer(Adam optimizer)    #계산한 gradient로부터 weight를 업데이트
7. define metric for model performance(accuracy)    #학습모델의 성능을 판단하는 지표 설정
8. (optional) make checkpoing for saving    #학습 결과를 임시로 저장(선택)
9. train and validate neural network model
"""

# 0.import libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

#print(tf.__version__)    #2.1.0
#print(keras.__version__)    #2.2.4-tf

# 1.hyper parameters setting
learning_rate = 0.001
training_epochs = 15
batch_size = 100

tf.random.set_seed(777)

# 1.create checkpoint directory
cur_dir = os.getcwd()
ckpt_dir_name = 'checkpoints'
model_dir_name = 'minst_cnn_seq'

checkpoint_dir = os.path.join(cur_dir, ckpt_dir_name, model_dir_name)
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, model_dir_name)

# 2.load data
mnist = keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 2.datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    
    
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.    #[0,1]로 scaling
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)    #4차원으로 만들기 위해 비어있는 channel(마지막) 추가
    
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)    #one hot encoding
    
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)    #data를 batch size만큼 잘라서 전달

# 3.model function
def create_model():    #함수 형태로 네트워크 구성
    model = keras.Sequential()    #API 선언
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation=tf.nn.relu, padding='SAME', 
                                  input_shape=(28, 28, 1)))    #첫번째 layer에만 input shape 입력(height*width*channel)
    model.add(keras.layers.MaxPool2D(padding='SAME'))    #size 2*2, stride 2는 default이므로 생략
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation=tf.nn.relu, padding='SAME'))
    model.add(keras.layers.MaxPool2D(padding='SAME'))
    model.add(keras.layers.Flatten())    #fully connected layer 진입을 위해 결과값을 일렬로 펴줌
    model.add(keras.layers.Dense(256, activation=tf.nn.relu))   #dense layer
    model.add(keras.layers.Dropout(0.4))    #dense layer parameter 수가 많으므로 dropout 적용
    model.add(keras.layers.Dense(10))    #layer를 순서대로 하나씩 추가
    return model
model = create_model()
model.summary()    #model에 대한 정보 표출
""" 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 128)         73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               524544    
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 619,786
Trainable params: 619,786
Non-trainable params: 0
_________________________________________________________________
 """

# 4.loss function
@tf.function

#    WARNING:tensorflow:Entity <function loss_fn at 0x000002A0B6640EA0> could not be transformed and will be staged without change. 
#    Error details can be found in the logs when running with the env variable AUTOGRAPH_VERBOSITY >= 1. 
#    Please report this to the AutoGraph team. 
#    Cause: Unexpected error transforming <function loss_fn at 0x000002A0B6640EA0>. 
#    If you believe this is due to a bug, please set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output when filing the bug report. 
#    Caused by: Bad argument number for Name: 3, expecting 4
#라고 뜨는 경우 있음. @tf.function 지우고 돌리면 Warning 없이 작동할 수도?

def loss_fn(model, images, labels):    #labels=정답
    logits = model(images, training=True)    #training True로 설정하면 model의 dropout layer에서 dropout이 적용
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(    #softmax도 결과 확률 총합을 1로 만들고, from_logits=True도 경과 총합을 1로 만드므로 둘 중 하나만 활성화(여기서는 softmax 없으므로 True로 설정)
        y_pred=logits, y_true=labels, from_logits=True))    #softmax cross entropy로 loss function 계산
    return loss   

# 5.calculate gradient
@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)    #back propagation(model을 거꾸로 거슬러 올라가며 gradient 계산)

# 7.calculate model's accuracy
@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)    #training이 아니므로 기능 끄기
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))    #정답(label)과 계산값(logit)을 비교
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# 6.AdamOptimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 8.create checkpoint for save
checkpoint = tf.train.Checkpoint(cnn=model)

# 9.training 함수 정의
@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# 9.model 학습
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):    # 1 training epoch가 1 cycle
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0    
    
    for images, labels in train_dataset:    # training epoch 내에서 1 batch size가 1 cycle
        train(model, images, labels)
        #grads = grad(model, images, labels)                
        #optimizer.apply_gradients(zip(grads, model.variables))    #training 함수 정의 부분에서 정의한 부분
        loss = loss_fn(model, images, labels)
        acc = evaluate(model, images, labels)    #학습 중간에 loss, acc 계산
        avg_loss = avg_loss + loss
        avg_train_acc = avg_train_acc + acc    #평균 계산
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    for images, labels in test_dataset:      # 1 epoch가 끝나면 test dataset을 넣고 모델 검증 
        acc = evaluate(model, images, labels)        
        avg_test_acc = avg_test_acc + acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
          'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))    #epoch 끝날 때마다 training, test 정확도 출력
    
    checkpoint.save(file_prefix=checkpoint_prefix)    #epoch 끝날 때마다 checkpoint save

print('Learning Finished!')
""" 
Learning started. It takes sometime.
Epoch: 1 loss = 0.16613261 train accuracy =  0.9571 test accuracy =  0.9846
Epoch: 2 loss = 0.04001159 train accuracy =  0.9903 test accuracy =  0.9908
Epoch: 3 loss = 0.02658202 train accuracy =  0.9933 test accuracy =  0.9921
Epoch: 4 loss = 0.01917950 train accuracy =  0.9958 test accuracy =  0.9905
Epoch: 5 loss = 0.01426240 train accuracy =  0.9967 test accuracy =  0.9929
Epoch: 6 loss = 0.01188609 train accuracy =  0.9972 test accuracy =  0.9934
Epoch: 7 loss = 0.00860970 train accuracy =  0.9982 test accuracy =  0.9931
Epoch: 8 loss = 0.00767121 train accuracy =  0.9984 test accuracy =  0.9935
Epoch: 9 loss = 0.00613560 train accuracy =  0.9988 test accuracy =  0.9912
Epoch: 10 loss = 0.00526856 train accuracy =  0.9987 test accuracy =  0.9928
Epoch: 11 loss = 0.00428856 train accuracy =  0.9993 test accuracy =  0.9936
Epoch: 12 loss = 0.00360134 train accuracy =  0.9992 test accuracy =  0.9934
Epoch: 13 loss = 0.00293311 train accuracy =  0.9993 test accuracy =  0.9914
Epoch: 14 loss = 0.00266282 train accuracy =  0.9993 test accuracy =  0.9921
Epoch: 15 loss = 0.00215506 train accuracy =  0.9995 test accuracy =  0.9938
Learning Finished!
 """

#dropout(DNN 부분)
#training 과정에서 모든 node를 활성화하지 않고 일부 node를 끄고 학습시켜 overfitting 방지
def dropout(rate) :
    return tf.keras.layers.Dropout(rate)    #rate=node를 비활성화하는 비율(0~1 값)

#batch normalization(DNN 부분)
#training 과정에서 layer마다 normalization을 다시 해서 data의 분포가 훼손되는 것을 방지
def batch_norm() :
    return tf.keras.layers.BatchNormalization()
model.add(batch_norm())    #layer>norm>activation

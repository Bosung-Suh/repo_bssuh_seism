""" How to make class
1. 무작정 만드는 것보다 클래스로 만든 객체를 중심으로 
어떤 식으로 동작하게 할 것인지 미리 구상을 한 후에 생각한 것들을 하나씩 해결하면서 완성해 나가는 것이 좋다.

2. 먼저 객체 생성 기능
class aaaa:
    pass    #object type: <class '__main__.aaaa'>

3. 객체에 숫자 지정(객체변수)
class FourCal:
    def setdata(self, first, second):
        self.first = first
        self.second = second
self 매개변수에는 setdata 메서드를 호출한 객체 a가 자동으로 전달.
- 클래스로 만든 객체의 객체변수는 다른 객체의 객체변수에 상관없이 독립적인 값을 유지한다. 

- 생성자(Constructor)
    이렇게 객체에 초깃값을 설정해야할 필요가 있을 때는 setdata와 같은 메서드를 호출하여 초깃값을 설정하기보다는 
    생성자를 구현하는 것이 안전한 방법이다(초깃값 설정 안 해도 자동으로 설정되어 오류 미발생). 생성자란 객체가 생성될 때 자동으로 호출되는 메서드를 의미한다(__init__).

4. 원하는 기능을 함수로 작성

- 함수 선언 내부에 *arg로 넣으면 갯수에 상관없이 모든 입력값을 튜플로 변환해 입력함 
- 키워드 매개변수 **kwargs : aaaa=1111로 입력된 설정들이 딕셔너리로 변환되어 입력됨

class Calculator:
    def __init__(self):
        self.result = 0    #클래스 개시

    def add(self, num):    #클래스의 메서드
        self.result += num
        return self.result
    def sub(self, num):
        self.result -= num
        return self.result

cal1 = Calculator()    #실제 객체 생성
cal2 = Calculator()

print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))

- 클래스 상속
    class bbbb(aaaa):    #aaaa클래스의 기능을 bbbb클래스에서도 사용
    기존 클래스를 변경하지 않고 기능을 추가하거나 기존 기능을 변경하려고 할 때 사용한다.(기능 확장)
- 메소드 오버라이딩
    부모 클래스의 메소드를 동일한 이름의 메소드로 재정의(덮어쓰기)
- 클래스 변수(클래스 안에 변수 선언)
class Family:
    lastname = "김"    #Family.lastname, 객체.lastname으로 호출 가능
    클래스 변수는 대입 연산자로 덮어쓰기 가능, 클래스의 모든 객체에 공유됨.
    이 상태에서 특정 객체의 클래스 변수를 수정하면(a.lastname=1111) 클래스 변수가 수정되는 것이 아니라 동일한 이름의 객체변수가 새로 생성됨.
    변수를 호출하면 클래스 자체나 다른 객체에서는 클래스 변수가 호출되고, 덮어쓴 객체에서는 객체 변수가 호출됨.
"""
"""
모듈 : 함수, 변수, 클래스 등을 모아놓은 파일. 다른 파일에서 불러와 사용 가능.
파이썬 확장자 .py로 만든 파이썬 파일은 모두 모듈이다
- 대화형 인터프리터의 경우 반드시 모듈을 저장한 디렉터리에서 진행해야 모듈을 읽을 수 있다.
다른 파일에서 사용하는 경우 import 파일명 (.py는 제거). 이 경우에도 모듈 파일과 import 받는 파일이 동일한 디렉터리에 있어야 함.
- 불러온 모듈의 함수를 사용하려면 모듈.함수명으로 호출
모듈명 없이 함수명 바로 쓰려면 from 모듈명 import 모듈함수1, 모듈함수2, ... 으로 호출(전부 다 호출하고 싶으면 정규표현식 *)
- __name__ : 직접 실행 시 __main__, import하는 경우 모듈명이 저장되는 변수명
대화형 인터프리터에서 모듈을 import하면 모듈 파일 자체가 실행되어 print 명령어 등의 결과값이 출력됨. 
명령어가 실행되는 부분을 조건문 if __name__ == "__main__": 으로 감싸면 
파일을 직접 열어서 실행할 때는 출력되고, 다른 파일에서 호출할 때는 출력 안 되게 할 수 있음.
- 모듈명.변수명, 모듈명.클래스명으로 변수값, 클래스 호출 가능. 

- 다른 디렉터리에 존재하는 모듈을 불러오는 방법
1. sys.path.append
2. PYTHONPATH 환경변수 사용

"""
#conda env python version 3.7.12
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class ModelConfig:
    batch_size = 10    # number of training examples in single iteration
    depths = 5
    filters_root = 8
    #kernel_size = [7, 1]
    #pool_size = [4, 1]
    #dilation_rate = [1, 1]
    #class_weights = [1.0, 1.0, 1.0]
    loss_type = "cross_entropy"
    weight_decay = 0.0
    optimizer = "adam"
    momentum = 0.9
    learning_rate = 0.01
    decay_step = 1e9
    decay_rate = 0.9
    dropout_rate = 0.0
    summary = True
  
    X_shape = [None, 1, 6000, 3]
    n_channel = X_shape[-1]
    #Y_shape = [3000, 1, 3]
    #n_class = Y_shape[-1]
  
def crop_and_concat(net1, net2):
    # net1_shape = net1.get_shape().as_list()
    # net2_shape = net2.get_shape().as_list()
    # # print(net1_shape)
    # # print(net2_shape)
    # # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
    # offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    # size = [-1, net1_shape[1], net1_shape[2], -1]
    # net2_resize = tf.slice(net2, offsets, size)
    # return tf.concat([net1, net2_resize], 3)

    ## dynamic shape
    chn1 = net1.get_shape().as_list()[-1]
    chn2 = net2.get_shape().as_list()[-1]
    net1_shape = tf.shape(net1)
    net2_shape = tf.shape(net2)
    # print(net1_shape)
    # print(net2_shape)
    # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
    offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)

    out = tf.concat([net1, net2_resize], 3)
    out.set_shape([None, None, None, chn1+chn2])

    return out 

    # else:
    #     offsets = [0, (net1_shape[1] - net2_shape[1]) // 2, (net1_shape[2] - net2_shape[2]) // 2, 0]
    #     size = [-1, net2_shape[1], net2_shape[2], -1]
    #     net1_resize = tf.slice(net1, offsets, size)
    #     return tf.concat([net1_resize, net2], 3)


def crop_only(net1, net2):
    net1_shape = net1.get_shape().as_list()
    net2_shape = net2.get_shape().as_list()
    # print(net1_shape)
    # print(net2_shape)
    # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
    offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    #return tf.concat([net1, net2_resize], 3)
    return net2_resize


class UNet:
    def __init__(self, config=ModelConfig(),):
        self.depths = config.depths
        self.filters_root = config.filters_root
        self.kernel_size = config.kernel_size
        self.dilation_rate = config.dilation_rate
        self.pool_size = config.pool_size
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.class_weights = config.class_weights
        self.batch_size = config.batch_size
        self.loss_type = config.loss_type
        self.weight_decay = config.weight_decay
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum
        self.global_step = tf.compat.v1.get_variable(name="global_step", initializer=0, dtype=tf.int32)
        self.summary_train = []
        self.summary_valid = []        
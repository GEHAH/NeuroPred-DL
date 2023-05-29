from tensorflow.keras.layers import (Input, Dense,Activation, BatchNormalization, Conv1D, Conv2D,MaxPooling1D, MaxPooling2D, LSTM, GRU, Embedding, Bidirectional,
                         Concatenate,Dropout, Embedding,Convolution1D, Flatten,Layer)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
# Performance:
from sklearn.metrics import (confusion_matrix, classification_report, matthews_corrcoef, precision_score)
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split)
# import pydot_ng as pydot
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical as labelEncoding   # Usages: Y = labelEncoding(Y, dtype=int)
from tensorflow.keras.utils import plot_model,model_to_dot

from sklearn.metrics import (confusion_matrix, classification_report, matthews_corrcoef, precision_score, roc_curve, auc)
from sklearn.model_selection import (StratifiedKFold, KFold, train_test_split)
from layers import MultiHeadAttention,Attention,AttLayer
from scipy import interp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,matthews_corrcoef,confusion_matrix,roc_curve, precision_recall_curve
import numpy as np
my_seed = 42
np.random.seed(my_seed)
import random
random.seed(my_seed)
import tensorflow as tf
tf.random.set_seed(my_seed)


def embedding_model():
    in_put = Input(shape=(100,))
    x = Embedding(input_dim=21, output_dim=100, input_length=100)(in_put)
    a = Convolution1D(128, 3, activation='relu', padding='valid')(x)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])


def one_hot_model():
    in_put = Input(shape=(100, 20))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])


def our_model():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])

def CNN_model():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    d = Dropout(0.2)(c)
    # d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])

def CNN_LSTM():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    # d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    d = LSTM(128,return_sequences=True)(c)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])

def CNN_GRU():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    # d = Bidirectional(LSTM(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    d = GRU(128,return_sequences=True)(c)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])

def CNN_BiGRU():
    in_put = Input(shape=(97, 150))
    a = Convolution1D(128, 3, activation='relu', padding='valid')(in_put)
    a = BatchNormalization()(a)
    #     a = MaxPooling1D(pool_size=3, strides=1,padding='valid')(a)
    b = Convolution1D(128, 3, activation='relu', padding='valid')(a)
    b = BatchNormalization()(b)
    c = Convolution1D(256, 3, activation='relu', padding='valid')(b)
    c = MaxPooling1D(pool_size=3, strides=1, padding='valid')(c)
    c = Dropout(0.2)(c)
    d = Bidirectional(GRU(128, return_sequences=True))(c)
    #     d = MultiHeadAttention(head_num=64, activation='relu', use_bias=True,
    #                                 return_multi_attention=False, name='Multi-Head-Attention')(d)
    # d = LSTM(128,return_sequences=True)(c)
    head = Flatten()(d)
    #     e = Dropout(0.5)(head)
    e = Dense(128, activation='relu', name='FC3')(head)
    e = Dropout(rate=0.5)(e)
    e = Dense(64, activation='relu', name='FC2')(e)

    e = Dense(32, activation='relu', name='FC4')(e)

    output = Dense(2, activation='softmax', name='Output')(e)

    return Model(inputs=[in_put], outputs=[output])
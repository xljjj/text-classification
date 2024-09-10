```python
pip install tensorflow-gpu==2.3.0
```


```python
import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.callbacks import ModelCheckpoint
import moxing as mox
import argparse
```


```python
from tensorflow.keras.layers import LSTM
```


```python
import tensorflow as tf
```


```python
# BASE_DIR为训练集根目录，这里设置为桶的dataset目录
BASE_DIR = './data'
```


```python
# 文本语料路径
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# 存储词向量到字典中
print('Indexing word vectors.')
print(TEXT_DATA_DIR)
embeddings_index = {}
with open(os.path.join(BASE_DIR, 'glove.6B.100d.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs
```

    Indexing word vectors.
    ./data/20_newsgroup



```python
#将每篇文章的文件名和标签进行存储
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)
print('Found %s texts.' % len(texts))
```

    Found 19997 texts.



```python
#分词
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
joblib.dump(tokenizer, 'token_result.pkl')
```




    ['token_result.pkl']




```python
#数据打乱和划分
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(data)
labels = to_categorical(np.asarray(labels))
print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
print(data)
labels = labels[indices]
print(labels)
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
print(data.shape[0])
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
```

    [[  58  576    3 ...    4  930 2050]
     [ 221   31  972 ... 2932  552  324]
     [   0    0    0 ...    3  316 5816]
     ...
     [   0    0    0 ...   71  197  514]
     [   0    0    0 ... 2113 1618 9557]
     [   0    0    0 ...    3    1 2703]]
    [[1. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]]
    Shape of data tensor: (19997, 1000)
    Shape of label tensor: (19997, 20)
    [[    0     0     0 ...     4  1636   453]
     [    0     0     0 ... 13710     6 14246]
     [    0     0     0 ...  3554   344  2182]
     ...
     [    0     0     0 ...  5734 11553    26]
     [    0     0     0 ...  2433  3662   813]
     [    0     0     0 ...   439  4032  9247]]
    [[0. 0. 0. ... 0. 1. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 1. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]
    19997



```python
#数据降维
print('Preparing embedding matrix.')
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        # 从预训练模型的词向量到语料库的词向量映射
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
```

    Preparing embedding matrix.
    Training model.



```python
print(embedded_sequences.shape)
#LSTM-CNN模型
#首先通过Embedding Layer将单词转化为词向量
#再输入LSTM进行语义特征提取
#下一步将LSTM的输出作为CNN的输入
#进行进一步的特征提取
#最后得到分类结果
```

    (None, 1000, 100)



```python
#LSTM层
lstm_layer=LSTM(units=256,batch_size=128)
lstm_sequences=lstm_layer(embedded_sequences)
print(lstm_sequences.shape)
```

    (None, 256)



```python
# 卷积要求输入为3维
lstm_sequences=tf.reshape(lstm_sequences,shape=[-1,256,1])
print(lstm_sequences.shape)
```

    (None, 256, 1)



```python
#CNN层
cnn_layer=Conv1D(filters=256,kernel_size=129,padding='valid', activation=tf.nn.relu)
cnn_sequences=cnn_layer(lstm_sequences)
print(cnn_sequences.shape)
```

    (None, 128, 256)



```python
#MaxPooling层
maxPooling_layer=MaxPooling1D()
maxPooling_sequences=maxPooling_layer(cnn_sequences)
print(maxPooling_sequences.shape)
```

    (None, 64, 256)



```python
#展平为1维
maxPooling_sequences=tf.reshape(maxPooling_sequences,shape=[-1,64*256])
maxPooling_sequences=tf.nn.dropout(maxPooling_sequences,rate=0.11)
print(maxPooling_sequences.shape)
```

    (None, 16384)



```python
dense_layer1=Dense(units=4096,activation=tf.nn.relu)
dense_sequences1=dense_layer1(maxPooling_sequences)
print(dense_sequences1.shape)
```

    (None, 4096)



```python
dense_layer2=Dense(units=1024,activation=tf.nn.relu)
dense_sequences2=dense_layer2(dense_sequences1)
print(dense_sequences2.shape)
```

    (None, 1024)



```python
output_layer=Dense(units=labels.shape[1],activation='softmax')
outputs=output_layer(dense_sequences2)
print(outputs.shape)
```

    (None, 20)



```python
model = Model(inputs = sequence_input, outputs = outputs)
model.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            [(None, 1000)]       0                                            
    __________________________________________________________________________________________________
    embedding_1 (Embedding)         (None, 1000, 100)    2000000     input_2[0][0]                    
    __________________________________________________________________________________________________
    lstm (LSTM)                     (None, 256)          365568      embedding_1[0][0]                
    __________________________________________________________________________________________________
    tf_op_layer_Reshape (TensorFlow [(None, 256, 1)]     0           lstm[0][0]                       
    __________________________________________________________________________________________________
    conv1d (Conv1D)                 (None, 128, 256)     33280       tf_op_layer_Reshape[0][0]        
    __________________________________________________________________________________________________
    max_pooling1d (MaxPooling1D)    (None, 64, 256)      0           conv1d[0][0]                     
    __________________________________________________________________________________________________
    tf_op_layer_Reshape_1 (TensorFl [(None, 16384)]      0           max_pooling1d[0][0]              
    __________________________________________________________________________________________________
    tf_op_layer_Shape (TensorFlowOp [(2,)]               0           tf_op_layer_Reshape_1[0][0]      
    __________________________________________________________________________________________________
    tf_op_layer_RandomUniform (Tens [(None, 16384)]      0           tf_op_layer_Shape[0][0]          
    __________________________________________________________________________________________________
    tf_op_layer_GreaterEqual (Tenso [(None, 16384)]      0           tf_op_layer_RandomUniform[0][0]  
    __________________________________________________________________________________________________
    tf_op_layer_Mul (TensorFlowOpLa [(None, 16384)]      0           tf_op_layer_Reshape_1[0][0]      
    __________________________________________________________________________________________________
    tf_op_layer_Cast (TensorFlowOpL [(None, 16384)]      0           tf_op_layer_GreaterEqual[0][0]   
    __________________________________________________________________________________________________
    tf_op_layer_Mul_1 (TensorFlowOp [(None, 16384)]      0           tf_op_layer_Mul[0][0]            
                                                                     tf_op_layer_Cast[0][0]           
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 4096)         67112960    tf_op_layer_Mul_1[0][0]          
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1024)         4195328     dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 20)           20500       dense_1[0][0]                    
    ==================================================================================================
    Total params: 73,727,636
    Trainable params: 71,727,636
    Non-trainable params: 2,000,000
    __________________________________________________________________________________________________



```python
model.compile(loss=tf.losses.categorical_crossentropy,optimizer=tf.optimizers.RMSprop(learning_rate=0.005),metrics=['acc'])
```


```python
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(x_val, y_val))
```

    Epoch 1/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.5375 - acc: 0.8175 - val_loss: 1.4660 - val_acc: 0.6782
    Epoch 2/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.4883 - acc: 0.8338 - val_loss: 1.4815 - val_acc: 0.7057
    Epoch 3/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.4690 - acc: 0.8498 - val_loss: 1.6598 - val_acc: 0.6992
    Epoch 4/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.4190 - acc: 0.8633 - val_loss: 1.4644 - val_acc: 0.6979
    Epoch 5/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.3975 - acc: 0.8707 - val_loss: 1.5808 - val_acc: 0.7082
    Epoch 6/10
    125/125 [==============================] - 20s 157ms/step - loss: 0.3697 - acc: 0.8773 - val_loss: 1.6560 - val_acc: 0.7214
    Epoch 7/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.3665 - acc: 0.8833 - val_loss: 2.0890 - val_acc: 0.7107
    Epoch 8/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.3529 - acc: 0.8904 - val_loss: 2.0226 - val_acc: 0.7147
    Epoch 9/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.3383 - acc: 0.8980 - val_loss: 1.8644 - val_acc: 0.7197
    Epoch 10/10
    125/125 [==============================] - 20s 158ms/step - loss: 0.3131 - acc: 0.8987 - val_loss: 2.4505 - val_acc: 0.7157



```python
# 先在虚拟机上保存模型，再将模型拷贝至桶的输出路径下。
Model_DIR = os.path.join(os.getcwd(), 'mytextcnn_model.h5')
model.save(Model_DIR)
print('Saved model to disk'+Model_DIR)
# 第二个参数需要根据实验者的桶路径修改
mox.file.copy_parallel(Model_DIR,'obs://nlp-textclassifier/output/mytextcnn_model.h5')
```

    Saved model to disk/home/ma-user/work/mytextcnn_model.h5


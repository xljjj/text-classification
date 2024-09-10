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
from tensorflow.keras.layers import LSTM
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='LSTM-CNN Example')
    parser.add_argument('--data_url', type=str, default=False,
                        help='path where the dataset is saved')
    parser.add_argument('--train_url', type=str, default=False, help='model path')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # BASE_DIR为训练集根目录，这里设置为桶的dataset目录
    BASE_DIR = args.data_url

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

    #分词
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    joblib.dump(tokenizer, 'token_result.pkl')

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

    print(embedded_sequences.shape)
    #LSTM-CNN模型
    #首先通过Embedding Layer将单词转化为词向量
    #再输入LSTM进行语义特征提取
    #下一步将LSTM的输出作为CNN的输入
    #进行进一步的特征提取
    #最后得到分类结果

    #LSTM层
    lstm_layer=LSTM(units=256,batch_size=128)
    lstm_sequences=lstm_layer(embedded_sequences)
    print(lstm_sequences.shape)

    # 卷积要求输入为3维
    lstm_sequences=tf.reshape(lstm_sequences,shape=[-1,256,1])
    print(lstm_sequences.shape)

    #CNN层
    cnn_layer=Conv1D(filters=256,kernel_size=129,padding='valid', activation=tf.nn.relu)
    cnn_sequences=cnn_layer(lstm_sequences)
    print(cnn_sequences.shape)

    #MaxPooling层
    maxPooling_layer=MaxPooling1D()
    maxPooling_sequences=maxPooling_layer(cnn_sequences)
    print(maxPooling_sequences.shape)

    #展平为1维
    maxPooling_sequences=tf.reshape(maxPooling_sequences,shape=[-1,64*256])
    maxPooling_sequences=tf.nn.dropout(maxPooling_sequences,rate=0.11)
    print(maxPooling_sequences.shape)

    dense_layer1=Dense(units=4096,activation=tf.nn.relu)
    dense_sequences1=dense_layer1(maxPooling_sequences)
    print(dense_sequences1.shape)

    dense_layer2=Dense(units=1024,activation=tf.nn.relu)
    dense_sequences2=dense_layer2(dense_sequences1)
    print(dense_sequences2.shape)

    output_layer=Dense(units=labels.shape[1],activation='softmax')
    outputs=output_layer(dense_sequences2)
    print(outputs.shape)

    model = Model(inputs = sequence_input, outputs = outputs)
    model.summary()

    model.compile(loss=tf.losses.categorical_crossentropy,optimizer=tf.optimizers.RMSprop(learning_rate=0.005),metrics=['acc'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=10,
                        validation_data=(x_val, y_val))

    # 保存模型与适配 ModelArts 推理模型包规范
    if args.save_model:

        # 在 train_url 训练参数对应的路径内创建 model 目录
        model_path = os.path.join(args.train_url, 'model')
        os.makedirs(model_path, exist_ok = True)

        # 按 ModelArts 推理模型包规范，保存模型到 model 目录内
        model.save(os.path.join(model_path),'mytextcnn_model.h5')

        # 拷贝推理代码与配置文件到 model 目录内
        the_path_of_current_file = os.path.dirname(__file__)
        mox.file.copy_parallel(os.path.join(the_path_of_current_file, 'infer/customize_service.py'), os.path.join(model_path, 'customize_service.py'))
        mox.file.copy_parallel(os.path.join(the_path_of_current_file, 'infer/config.json'), os.path.join(model_path, 'config.json'))

if __name__ == '__main__':
    main()
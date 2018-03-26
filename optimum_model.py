from plotly.graph_objs import *
from plotly.offline import plot

import numpy as np
import os
import itertools

from keras.models import load_model, Sequential
from keras.layers import Flatten, MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout
from keras.optimizers import Nadam, SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import CSVLogger, EarlyStopping

seed = 97
np.random.seed(seed)
num_labels = 3

max_test_size = 1200
max_valid_size = 1200
max_training_size = 4600

pixel_depth = 255.0
im_size_1 = 7
im_size_2 = 7
im_size_3 = 1024
image_size = im_size_1 * im_size_2 * im_size_3


def load_pictures(folder, max_num, kind, dataType):
    dataset = np.ndarray(shape=(max_num, image_size), dtype=np.float32)
    labels = np.ndarray(max_num, dtype=np.int32)
    image_index = 0
    print(folder)

    def res(x):
        # return int(str(x).split('.')[0])  # FOR USUAL
        return int(str(x)[0:str(x).find('_')])  # FOR TURNED

    ar = np.array(sorted(os.listdir(folder), key=res))

    start = 0
    if dataType == 1:
        start = max_training_size

    if dataType == 2:
        start = max_training_size + max_test_size

    zip_ar = np.array(list(zip(np.arange(max_num), ar[start:start + max_num])))

    def func(image_index):
        image_index = int(image_index)
        im = zip_ar[image_index][1]
        image_file = os.path.join(folder, im)
        img = np.load(image_file)
        dataset[image_index, :] = img
        labels[image_index] = kind
        return 0

    new_arange = np.arange(max_num).reshape((-1, 1))
    np.apply_along_axis(func, 1, np.array(new_arange))

    return dataset, labels, image_index


def load_dataset(folder, max_num, dataType):
    dataset_0, labels_0, image_index_0 = load_pictures(folder + 'Control(sharp)(turn)', max_num, 0, dataType)
    dataset_1, labels_1, image_index_1 = load_pictures(folder + '01Taxol(sharp)(turn)', max_num, 1, dataType)
    dataset_2, labels_2, image_index_2 = load_pictures(folder + '1Taxol(sharp)(turn)', max_num, 2, dataType)
    dataset = np.concatenate((dataset_0, dataset_1, dataset_2), axis=0)
    labels = np.concatenate((labels_0, labels_1, labels_2), axis=0)

    return dataset, labels


train_dataset, train_labels = load_dataset('MobileNetPretrainedData/', max_training_size, 0)
test_dataset, test_labels = load_dataset('MobileNetPretrainedData/', max_test_size, 1)
valid_dataset, valid_labels = load_dataset('MobileNetPretrainedData/', max_valid_size, 2)

train_dataset = train_dataset.reshape(train_dataset.shape[0], im_size_1, im_size_2, im_size_3)
test_dataset = test_dataset.reshape(test_dataset.shape[0], im_size_1, im_size_2, im_size_3)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0], im_size_1, im_size_2, im_size_3)

train_dataset = train_dataset.astype('float32')
test_dataset = test_dataset.astype('float32')
valid_dataset = valid_dataset.astype('float32')

train_labels = np_utils.to_categorical(train_labels, num_labels)
test_labels = np_utils.to_categorical(test_labels, num_labels)
valid_labels = np_utils.to_categorical(valid_labels, num_labels)

pool_size = (2, 2)
kernel_size1 = (2, 2)
kernel_size2 = (2, 2)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=100, verbose=0, mode='min')
early_stop2 = EarlyStopping(monitor='val_acc', min_delta=0.0002, patience=100, verbose=0, mode='max')

kern1 = [(2, 2)]
dropout = [0.50]
l2_lambda = [(2 * 1e-2, 2 * 1e-2, 2 * 1e-2, 2 * 1e-2)]
depth = [(16, 8)]
hidden_number = [(256, 64)]
optimizer = [Nadam(lr=0.0002)]  # , 'adadelta', 'rmsprop']

params = list(itertools.product(kern1, dropout, l2_lambda, depth, hidden_number, optimizer))
for_shuffle = np.array(params)
np.random.shuffle(for_shuffle)
params = list(for_shuffle)

folder = 'Logs'
if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists('TmpModels'):
    os.makedirs('TmpModels')
model = 1

for i in range(len(params)):
    csv_logger = CSVLogger(folder + '/training' + str(i) + '.log')
    cur = params[i]
    kern = cur[0]
    drop = cur[1]
    lamb = cur[2]
    dep = cur[3]
    hid = cur[4]
    opt = cur[5]
    del model

    model = Sequential()
    model.add(Conv2D(dep[0], (kern[0], kern[1]), padding='valid', input_shape=(im_size_1, im_size_2, im_size_3),
                     activation='relu', W_regularizer=l2(lamb[0])))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(drop))
    model.add(Conv2D(dep[1], (kernel_size2[0], kernel_size2[1]), padding='same',
                     activation='relu', W_regularizer=l2(lamb[1])))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(drop))

    model.add(Flatten())

    model.add(Dense(1024, W_regularizer=l2(lamb[2]), activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(hid[0], W_regularizer=l2(lamb[2]), activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(hid[1], W_regularizer=l2(lamb[3]), activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # model = load_model('TmpModels/conv_tmp_model.h5')

    history = model.fit(train_dataset, train_labels, nb_epoch=100, batch_size=32, verbose=2,
                        validation_data=(valid_dataset, valid_labels),
                        callbacks=[csv_logger, early_stop, early_stop2])

    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    train_loss = history.history['loss']

    scores = model.evaluate(test_dataset, test_labels, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save('TmpModels/conv_tmp_model.h5')

    data = []
    data.append(
        Scatter(x=len(train_loss), y=train_loss, name="train loss", text='train loss', marker=dict(color='blue')))
    data.append(Scatter(x=len(train_loss), y=val_loss, name='validation loss', text='validation loss',
                        marker=dict(color='orange')))
    data.append(Scatter(x=len(train_loss), y=train_acc, name='train accuracy', text='train accuracy',
                        marker=dict(color='red')))
    data.append(Scatter(x=len(train_loss), y=val_acc, name='validation accuracy', text='validation accuracy',
                        marker=dict(color='green')))
    plot(Figure(data=data, layout=Layout()), filename=('tmp' + str(i + 1) + '.html'))

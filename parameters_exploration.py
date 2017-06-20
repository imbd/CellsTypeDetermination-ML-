import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import itertools

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Nadam, SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import CSVLogger, EarlyStopping

seed = 7
np.random.seed(seed)

num_labels = 2

max_test_size = 1500
max_valid_size = 2000
max_training_size = 7000

# max_test_size = 4000
# max_valid_size = 2000
# max_training_size = 4000

pixel_depth = 255.0
image_size = 100


def load_pictures(folder, max_num, kind, dataType):
    dataset = np.ndarray(shape=(max_num, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(max_num, dtype=np.int32)
    image_index = 0
    print(folder)

    def res(x):
        return int(str(x)[str(x).rfind('_') + 1:-4])

    ar = np.array(sorted(os.listdir(folder), key=res))
    start = 0
    if dataType == 1:
        start = max_training_size

    if dataType == 2:
        start = max_training_size + max_test_size

    zip_ar = np.array(list(zip(np.arange(max_num), ar[start:start + max_num])))
    print(zip_ar)
    print(zip_ar.shape)

    def func(image_index):
        image_index = int(image_index)
        if image_index % 500 == 0:
            print('in f', image_index)
        image = zip_ar[image_index][1]
        image_file = os.path.join(folder, image)
        image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
        dataset[image_index, :, :] = image_data[:, :]
        labels[image_index] = kind
        return 0

    new_arange = np.arange(max_num).reshape((-1, 1))
    print(new_arange.shape)
    np.apply_along_axis(func, 1, np.array(new_arange))

    print(dataset.shape)

    return dataset, labels, image_index


def load_dataset(folder, max_num, dataType):
    dataset_0, labels_0, image_index_0 = load_pictures(folder + '01Taxol(sharp)(100)(turn)', max_num, 0, dataType)
    dataset_1, labels_1, image_index_1 = load_pictures(folder + '1Taxol(sharp)(100)(turn)', max_num, 1, dataType)
    dataset = np.concatenate((dataset_0, dataset_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)
    print('Dataset: ', dataset.shape)

    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


print('TRAIN DATASETS')
train_dataset, train_labels = load_dataset('SAVE/', max_training_size, 0)
print(train_dataset.shape)

print('TEST DATASETS')
test_dataset, test_labels = load_dataset('SAVE/', max_test_size, 1)
print(test_dataset.shape)

print('VALID DATASETS')
valid_dataset, valid_labels = load_dataset('SAVE/', max_valid_size, 2)
print(valid_dataset.shape)

train_dataset = train_dataset.reshape(train_dataset.shape[0], image_size, image_size, 1)
test_dataset = test_dataset.reshape(test_dataset.shape[0], image_size, image_size, 1)
valid_dataset = valid_dataset.reshape(valid_dataset.shape[0], image_size, image_size, 1)
train_dataset = train_dataset.astype('float32')
test_dataset = test_dataset.astype('float32')
valid_dataset = valid_dataset.astype('float32')

train_labels = np_utils.to_categorical(train_labels, num_labels)
test_labels = np_utils.to_categorical(test_labels, num_labels)
valid_labels = np_utils.to_categorical(valid_labels, num_labels)

pool_size = (2, 2)
kernel_size1 = (2, 2)
kernel_size2 = (2, 2)

csv_logger = CSVLogger('PLOTS/training.log')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=162, verbose=0, mode='min')

'''kern1 = [(2,2), (5,5)]
dropout = [0.3, 0.5, 0.7]
l2_lambda = [(1e-5,1e-5,1e-10,1e-5), (1e-2,1e-2,0,1e-2), (1e-8,1e-8,0,1e-8)]
depth = [(1,1), (3,1), (4,2)]
hidden_number = [(512,32), (256,16), (512,256)]
optimizer = ['nadam', 'sgd', 'adadelta']'''

# '''
kern1 = [(2, 2)]
dropout = [0.6]
l2_lambda = [(1e-2, 1e-2, 0, 1e-2), (1e-8, 1e-8, 0, 1e-8)]
depth = [(1, 1), (3, 1), (4, 2)]
hidden_number = [(512, 32), (128, 16), (512, 256)]
optimizer = ['nadam', 'sgd', 'adadelta']
# '''

params = list(itertools.product(kern1, dropout, l2_lambda, depth, hidden_number, optimizer))
print(len(params))
for_shuffle = np.array(params)
np.random.shuffle(for_shuffle)
params = list(for_shuffle)
print(len(params))

i = 0
csv_logger = CSVLogger('Tmp/training' + str(i) + '.log')

print('Index: ', i)
print(str(params[i]))
cur = params[i]
kern = cur[0]
drop = cur[1]
lamb = cur[2]
dep = cur[3]
hid = cur[4]
opt = cur[5]
model = Sequential()
model.add(Convolution2D(dep[0], kern[0], kern[1], border_mode='valid', input_shape=(image_size, image_size, 1),
                        W_regularizer=l2(lamb[0])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(drop))
model.add(
    Convolution2D(dep[1], kernel_size2[0], kernel_size2[1], border_mode='same', W_regularizer=l2(lamb[1])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(drop))

model.add(Flatten())
model.add(Dense(hid[0], W_regularizer=l2(lamb[2])))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Dense(hid[1], W_regularizer=l2(lamb[3])))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# model = load_model('2conv_tmp_model.h5')

for j in range(1):
    print('Episode', j)
    if j > 0:
        model = load_model('2conv_tmp_model.h5')
    history = model.fit(train_dataset, train_labels, nb_epoch=15, batch_size=32, verbose=2,
                        validation_data=(valid_dataset, valid_labels), callbacks=[csv_logger, early_stop])

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.clf()
    val_acc = history.history['val_acc']
    print(val_acc[-1])
    scores = model.evaluate(test_dataset, test_labels, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    print()

    model.save('2conv_tmp_model.h5')

    scores = model.evaluate(valid_dataset, valid_labels, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print('Valid score:', scores[0])
    print('Valid accuracy:', scores[1])
    print()

    scores = model.evaluate(train_dataset, train_labels, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    print('Train score:', scores[0])
    print('Train accuracy:', scores[1])
    print()

print('THE END')
model.save('2conv_tmp_model.h5')
model.save('6tmp_1t01t_5.h5')

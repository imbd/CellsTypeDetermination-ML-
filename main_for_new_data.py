import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Nadam, SGD, Adam, RMSprop
from keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 7
np.random.seed(seed)

num_labels = 2

max_test_size = 4000
max_valid_size = 2000
max_training_size = 4000

# max_test_size = 400
# max_valid_size = 200
# max_training_size = 400


pixel_depth = 255.0
image_size = 300


def load_pictures(folder, max_num, kind, dataType):
    dataset = np.ndarray(shape=(max_num, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(max_num, dtype=np.int32)
    image_index = 0
    print(folder)

    def res(x):
        return int(str(x)[0:str(x).find('_')])

    ar = np.array(sorted(os.listdir(folder), key=res))
    start = 0
    if dataType == 1:
        start = max_training_size
    if dataType == 2:
        start = max_training_size + max_test_size
    zip_ar = np.array(list(zip(np.arange(max_num), ar[start:start + max_num])))

    print(zip_ar)

    def func(image_index):
        image_index = int(image_index)
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
    dataset_0, labels_0, image_index_0 = load_pictures(folder + '01Taxol(sharp)(300)(turn)', max_num, 0, dataType)
    dataset_1, labels_1, image_index_1 = load_pictures(folder + '1Taxol(sharp)(300)(turn)', max_num, 1, dataType)
    # dataset_1, labels_1, image_index_1 = load_pictures(folder + 'Control(sharp)(300)(turn)', max_num, 1, dataType)
    dataset = np.concatenate((dataset_0, dataset_1), axis=0)
    labels = np.concatenate((labels_0, labels_1), axis=0)
    print('Dataset: ', dataset.shape)

    return dataset, labels


print('TRAIN DATASETS')
train_dataset, train_labels = load_dataset('PreparedData/', max_training_size, 0)
print(train_dataset.shape)

print('TEST DATASETS')
test_dataset, test_labels = load_dataset('PreparedData/', max_test_size, 1)
print(test_dataset.shape)

print('VALID DATASETS')
valid_dataset, valid_labels = load_dataset('PreparedData/', max_valid_size, 2)
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
kernel_size1 = (5, 5)
kernel_size2 = (2, 2)

model = Sequential()

l2_lambda_1 = 1e-2
l2_lambda_2 = 1e-2
l2_lambda_3 = 0
l2_lambda_4 = 1e-2
drop = 0.70

model.add(
    Convolution2D(1, kernel_size1[0], kernel_size1[1], border_mode='valid', input_shape=(image_size, image_size, 1),
                  W_regularizer=l2(l2_lambda_1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(drop))
model.add(Convolution2D(1, kernel_size2[0], kernel_size2[1], border_mode='same', W_regularizer=l2(l2_lambda_2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(drop))

model.add(Flatten())
model.add(Dense(512, W_regularizer=l2(l2_lambda_3)))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Dense(64, W_regularizer=l2(l2_lambda_4)))
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Dense(num_labels, activation='softmax'))

optim = 'nadam'
# optim = Nadam(lr=0.00002)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

'''
episodes = 3
for j in range(episodes):
    print('Episode', j + 1)
    if j >= 0:
        model = load_model('2conv_tmp_model.h5')

    history = model.fit(train_dataset, train_labels, nb_epoch=1, batch_size=32, verbose=2,
                        validation_data=(valid_dataset, valid_labels))
    model.save('2conv_tmp_model.h5')
    val_loss = history.history['val_loss']
    if val_loss[-1] < 0.50:  # break if new good result found
        break
'''

model = load_model('2conv_tmp_model.h5')

scores = model.evaluate(train_dataset, train_labels, verbose=2)
print('TRAIN score:', scores[0])
print('TRAIN accuracy:', scores[1])

scores = model.evaluate(test_dataset, test_labels, verbose=2)
print('Test score:', scores[0])
print('Test accuracy:', scores[1])

scores = model.evaluate(valid_dataset, valid_labels, verbose=2)
print('VALID score:', scores[0])
print('VALID accuracy:', scores[1])

model.save('2conv_tmp_model.h5')
model.save('result.h5')

el_num = 8
diff_test_dataset = test_dataset[range(0, test_dataset.shape[0], el_num), :]
diff_test_labels = test_labels[range(0, test_labels.shape[0], el_num), :]
scores = model.evaluate(diff_test_dataset, diff_test_labels, verbose=2)
print('Real test score:', scores[0])
print('Real test accuracy:', scores[1])

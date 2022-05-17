import pandas as pd 
import numpy as np
import glob 
import os
import matplotlib.pyplot as plt
import time 
import random
import keras
import pathlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getData():
    # paths = glob.glob(r'./training_data/*.csv')
    file_list = os.listdir('training_data')
    datas = []
    max_dict = {'generation':-1.0, 'consumption':-1.0, 'month':12.0, 'day':31.0, 'hour':24.0}
    l = len(file_list)
    for i in range(l):
        # print('loading file {} ({}/{})'.format(file_list[i], i + 1, l))
        data = pd.read_csv('training_data/'+file_list[i], usecols=['generation', 'consumption'])
        time_stamp = pd.read_csv('training_data/'+file_list[i], usecols=['time'])

        data = gaussianBlur(data, 5)

        month = []
        day = []
        hour = []
        g_max, c_max = data['generation'].max(), data['consumption'].max()
        max_dict['generation'] = g_max if g_max > max_dict['generation'] else max_dict['generation']
        max_dict['consumption'] = c_max if c_max > max_dict['consumption'] else max_dict['consumption']
        for t in time_stamp['time']:
            month.append(float(str(t).split(' ')[0].split('-')[1]))
            day.append(float(str(t).split(' ')[0].split('-')[2]))
            hour.append(float(str(t).split(' ')[1].split(':')[0]))
        data['month'] = month
        data['day'] = day
        data['hour'] = hour
        datas.append(pd.DataFrame(data))

    return datas, max_dict

def gaussianBlur(data, ksize=3):
    # 跟數值的前後各2個值做 GaussianBlur
    assert ksize > 0 and ksize % 2 == 1, "ksize must be positive odd integer"
    # print('before', data)
    bias = (ksize - 1) // 2
    
    x = []
    for i in range(ksize):
        x.append([-bias + i])
    x = np.array(x)
    # print('x before exp', x)
    x = np.exp(-x**2)
    # print('x after exp', x)
    kernel = x / x.sum()
    pb, pe = [], []
    
    # padding begin
    for i in range(bias):
        pb.append(data.iloc[0].values)
    # padding end
    for i in range(bias):
        pe.append(data.iloc[-1].values)
    pb, pe = np.array(pb), np.array(pe)
    data_pad = np.vstack((pb, data.values, pe))

    for i in range(data.shape[0]):
        # tmp = data_pad[i:i+ksize]
        tmp = data_pad[i:i+ksize] * kernel
        for j in range(data.shape[1]):
            data.iloc[i,j] = np.sum(tmp[:,j])
    # print('after', data)
    # print('kernel', kernel)
    # print('data_pad', data_pad)
    # print('pb, pe',pb, pe)

    return data

def normalize(datas, max_dict):
    l = len(datas)
    for i in range(l):
        # print('normalizing {}/{}'.format(i + 1, l))
        datas[i]['generation'] = datas[i]['generation'].apply(lambda x: x / max_dict['generation'])
        datas[i]['consumption'] = datas[i]['consumption'].apply(lambda x: x / max_dict['consumption'])
        datas[i]['month'] = datas[i]['month'].apply(lambda x: x / max_dict['month'])
        datas[i]['day'] = datas[i]['day'].apply(lambda x: x / max_dict['day'])
        datas[i]['hour']= datas[i]['hour'].apply(lambda x: x / max_dict['hour'])
        # print(datas[i])
    return datas

def normalizeRecover(data, max_dict, key):
    data = data[:] * max_dict[key]
    return data

def setXYRawData(datas, ref_size, predict_size):
    x, y_con, y_gen = [], [], []
    for data in datas:
        for j in range(len(data) - ref_size -  predict_size):
            x.append(np.array(data.iloc[j: j + ref_size]))
            tmp1 = np.array(data.iloc[j + ref_size: j + ref_size + predict_size]['generation'])
            tmp2 = np.array(data.iloc[j + ref_size: j + ref_size + predict_size]['consumption'])
            y_con.append(tmp2)
            y_gen.append(tmp1)
    x, y_con, y_gen = np.array(x), np.array(y_con), np.array(y_gen)
    return x, y_con, y_gen

def shuffle(x, y):
    np.random.seed(int(time.time()))
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

# use LSTM model as training model
def buildManyToManyModel(in_shape, out_shape):
    model = Sequential()
    # LSTM input_shape 的參數設定為 (num_timesteps, num_features)
    # LSTM 第一個參數為 unit 表示一個 cell 內會有 64組的 gates，越多表越複雜 
    model.add(LSTM(64, input_shape=(in_shape[1], in_shape[2]), return_sequences = True))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularization
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularization
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularization
    model.add(TimeDistributed(Dense(units = 1)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(out_shape))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    model.summary()
    return model

def lossDump(dir, history):
    fig, ax = plt.subplots()
    ax.plot(history['val_loss'], label='val_loss')
    ax.plot(history['val_mse'], label='val_mse')
    ax.plot(history['loss'], label='loss')
    ax.plot(history['mse'], label='mse')
    ax.set_ylabel('result')
    ax.set_xlabel('epoch')
    ax.set_title('history')
    ax.legend()
    TABLE_NAME_PATH = dir + 'lossConsumption.png'
    pathlib.Path(TABLE_NAME_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(TABLE_NAME_PATH)

# code entry point

# constant
TRAIN_THRESH = 0.7
TEST_THRESH = 0.9
REF_SIZE = 24 * 7 # for one week
PREDICT_SIZE = 1 # for one hour
BATCH_SIZE = 128
PATIENCE = 10
EPOCH = 100
LEARNING_RATE = 0.01


# get a list of data from './training_data', select all the .csv file
datas, max_dict = getData()
# print(datas)
# print(max_dict)

# normalization
datas_norm = normalize(datas, max_dict)

# set x, y data
raw_x, raw_con_y, raw_gen_y = setXYRawData(datas_norm, REF_SIZE, PREDICT_SIZE)

# split train test
train_end = int(raw_x.shape[0] * TRAIN_THRESH)
test_end = int(raw_x.shape[0] * TEST_THRESH)

train_x, train_con_y = shuffle(raw_x[:train_end]         , raw_con_y[:train_end]         )
test_x , test_con_y  = shuffle(raw_x[train_end: test_end], raw_con_y[train_end: test_end])
val_x  , val_con_y   = shuffle(raw_x[test_end:]          , raw_con_y[test_end:]          )

train_x, train_gen_y = shuffle(raw_x[:train_end]         , raw_gen_y[:train_end]         )
test_x , test_gen_y  = shuffle(raw_x[train_end: test_end], raw_gen_y[train_end: test_end])
val_x  , val_gen_y   = shuffle(raw_x[test_end:]          , raw_gen_y[test_end:]          )

assert train_x.shape[0] == train_con_y.shape[0], "The lenth of con x and y need to be same."
assert train_x.shape[0] == train_gen_y.shape[0], "The lenth of gen x and y need to be same."

train = True
target_dir = 'lab/'
if train:
    # in this repo train.shape = (198205, 168, 5)
    # - 198205 約表示 : 5833(個小時) * 49(組帳號) * 0.7(的比例要拿來做 training)
    # - 168 表示 : 過去 24\*7 hour data 要拿來預測下一個階段的
    # - 5 表示 : 餵進去的 features 有`gen`、`con`、`month`、`day`、`hour`，共5種 features
    model_con = buildManyToManyModel(train_x.shape, PREDICT_SIZE) 
    model_gen = buildManyToManyModel(train_x.shape, PREDICT_SIZE) 

    early_stopping = EarlyStopping(monitor='val_mse', patience=PATIENCE, verbose=1, mode='min')

    history_con = model_con.fit(train_x, train_con_y, verbose=1, callbacks=[early_stopping],\
            validation_data=(val_x, val_con_y), batch_size=BATCH_SIZE, epochs=EPOCH)
    history_gen = model_con.fit(train_x, train_gen_y, verbose=1, callbacks=[early_stopping],\
            validation_data=(val_x, val_gen_y), batch_size=BATCH_SIZE, epochs=EPOCH)


    lossDump(target_dir, history_con.history)
    lossDump(target_dir, history_gen.history)

    TEST_SIZE = 24

    predict_con_ytrain = []
    answer_con_ytrain = []
    predict_con_ytest = []
    answer_con_ytest = []
    predict_gen_ytrain = []
    answer_gen_ytrain = []
    predict_gen_ytest = []
    answer_gen_ytest = []

    start_index = random.randint(0, train_x.shape[0] - TEST_SIZE)
    answer_con_y = train_con_y[start_index: start_index + TEST_SIZE]
    answer_gen_y = train_gen_y[start_index: start_index + TEST_SIZE]
    
    for (i, data) in enumerate(train_x[start_index: start_index + TEST_SIZE]):
        tmp_con = model_con.predict(np.array([data]))[0]
        tmp_con_recover = normalizeRecover(tmp_con, max_dict, 'consumption')
        tmp_con_ans = normalizeRecover(answer_con_y[i], max_dict, 'consumption')
        predict_con_ytrain.append(tmp_con_recover[0])
        answer_con_ytrain.append(tmp_con_ans[0])

        tmp_gen = model_gen.predict(np.array([data]))[0]
        tmp_gen_recover = normalizeRecover(tmp_gen, max_dict, 'generation')
        tmp_gen_ans = normalizeRecover(answer_gen_y[i], max_dict, 'generation')
        predict_gen_ytrain.append(tmp_gen_recover[0])
        answer_gen_ytrain.append(tmp_gen_ans[0])

    start_index = random.randint(0, test_x.shape[0] - TEST_SIZE)
    answer_con_y = test_con_y[start_index: start_index + TEST_SIZE]
    answer_gen_y = test_gen_y[start_index: start_index + TEST_SIZE]

    for (i, data) in enumerate(test_x[start_index: start_index + TEST_SIZE]):
        tmp_con = model_con.predict(np.array([data]))[0]
        tmp_con_recover = normalizeRecover(tmp_con, max_dict, 'consumption')
        tmp_con_ans = normalizeRecover(answer_con_y[i], max_dict, 'consumption')
        predict_con_ytest.append(tmp_con_recover[0])
        answer_con_ytest.append(tmp_con_ans[0])

        tmp_gen = model_con.predict(np.array([data]))[0]
        tmp_gen_recover = normalizeRecover(tmp_gen, max_dict, 'generation')
        tmp_gen_ans = normalizeRecover(answer_gen_y[i], max_dict, 'generation')
        predict_gen_ytest.append(tmp_gen_recover[0])
        answer_gen_ytest.append(tmp_gen_ans[0])


    fig, ax = plt.subplots(2, figsize=(20, 12))
    ax[0].plot(answer_con_ytrain, 'go-', label = 'train ans')
    ax[0].plot(predict_con_ytrain, 'bo-', label = 'train predict')
    ax[0].set_xlabel('index')
    ax[0].set_ylabel('consumption')
    ax[0].legend(loc = 'upper right')
    ax[1].plot(answer_con_ytest, 'go-', label = 'test ans')
    ax[1].plot(predict_con_ytest, 'bo-', label = 'test predict')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('consumption')
    ax[1].legend(loc = 'upper right')
    plt.savefig(target_dir + 'consumption_test.png')

    fig, ax = plt.subplots(2, figsize=(20, 12))
    ax[0].plot(answer_gen_ytrain, 'go-', label = 'train ans')
    ax[0].plot(predict_gen_ytrain, 'bo-', label = 'train predict')
    ax[0].set_xlabel('index')
    ax[0].set_ylabel('generation')
    ax[0].legend(loc = 'upper right')
    ax[1].plot(answer_gen_ytest, 'go-', label = 'test ans')
    ax[1].plot(predict_gen_ytest, 'bo-', label = 'test predict')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('generation')
    ax[1].legend(loc = 'upper right')
    plt.savefig(target_dir + 'generation_test.png')

    # save model
    save_name = target_dir + 'consumption-{}-{}.h5'.format(np.around(np.min(history_con.history['val_mse']), decimals=6), time.ctime(time.time()).replace(':','-'))
    model_con.save(save_name,save_format='h5')

    save_name = target_dir + 'generation-{}-{}.h5'.format(np.around(np.min(history_gen.history['val_mse']), decimals=6), time.ctime(time.time()).replace(':','-'))
    model_con.save(save_name,save_format='h5')

else :
    model = keras.models.load_model(target_dir + 'consumption-0.000363-Tue May 17 12-26-20 2022.h5')

    TEST_SIZE = 24

    predict_ytest = []
    answer_ytest = []

    start_index = random.randint(0, test_x.shape[0] - TEST_SIZE)
    test_con_ytest = test_con_y[start_index: start_index + TEST_SIZE]
    for (i, data) in enumerate(test_x[start_index: start_index + TEST_SIZE]):
        tmp = model.predict(np.array([data]))[0]
        tmp_recover = normalizeRecover(tmp, max_dict, 'consumption')
        tmp_ans = normalizeRecover(test_con_ytest[i], max_dict, 'consumption')
        # print('result', tmp_recover[0], tmp_ans[0])
        predict_ytest.append(tmp_recover[0])
        answer_ytest.append(tmp_ans[0])

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(answer_ytest, label = 'test ans')
    ax.plot(predict_ytest, label = 'test predict')
    ax.set_xlabel('index')
    ax.set_ylabel('consumption')
    ax.legend()
    plt.savefig(target_dir + 'consumption_test.png')

print('process complete.')
import pandas as pd 
import numpy as np
import glob 
import pathlib
import matplotlib.pyplot as plt
import time 
import random
import keras
import argparse

def gaussianBlur(data, ksize=3):
    assert ksize > 0 and ksize % 2 == 1, "ksize must be positive odd integer"
    bias = (ksize - 1) // 2
    
    x = []
    for i in range(ksize):
        x.append([-bias + i])
    x = np.array(x)
    x = np.exp(-x**2)
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
        tmp = data_pad[i:i+ksize] * kernel
        for j in range(data.shape[1]):
            data.iloc[i,j] = np.sum(tmp[:,j])

    return data

def normalize(data, max_dict):
    data['generation'] = data['generation'].apply(lambda x: x / max_dict['generation'])
    data['consumption'] = data['consumption'].apply(lambda x: x / max_dict['consumption'])
    data['month'] = data['month'].apply(lambda x: x / max_dict['month'])
    data['day'] = data['day'].apply(lambda x: x / max_dict['day'])
    data['hour']= data['hour'].apply(lambda x: x / max_dict['hour'])
    return data

def normalizeRecover(data, max_dict, key):
    return data * max_dict[key]

def setInputRawGC(pathG, pathC, max_dict):
    dataG = pd.read_csv(pathG, usecols=['time', 'generation'])
    dataC = pd.read_csv(pathC, usecols=['time', 'consumption'])
    timeInfo = pd.read_csv(pathC, usecols=['time'])
    data_raw = pd.merge(dataG, dataC, on='time')
    data_raw = pd.DataFrame(data_raw).drop(columns=['time'])
    data_raw = gaussianBlur(data_raw, 5)
    month = []
    day = []
    hour = []
    for t in timeInfo['time']:
        month.append(float(str(t).split(' ')[0].split('-')[1]))
        day.append(float(str(t).split(' ')[0].split('-')[2]))
        hour.append(float(str(t).split(' ')[1].split(':')[0]))
    data_raw['month'] = month
    data_raw['day'] = day
    data_raw['hour'] = hour
    print(data_raw)

    return data_raw

def timeInc(month, day, hour, inc = 1):
    dayDict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    thresh = 0.00001
    assert 1 <= month and month <= 12 + thresh, "month value error : {}".format(month)
    assert 1 <= day and day <= dayDict[month] + thresh, "day value error : {}".format(day)
    assert 0 <= hour and hour <= 23 + thresh, "hour value error : {}".format(hour)
    
    incD = (inc + hour) // 24
    hour = (inc + hour) - (incD * 24)

    while (incD + day) > dayDict[month]:
        incD -= (dayDict[month] - day + 1)
        month = month + 1 if month < 12 else 1
        day = 1
    day += incD

    return month, day, hour

def formatTime(month, day, hour):
    return '2018-{}-{} {}:00:00'.format(str(int(month)).zfill(2), str(int(day)).zfill(2), str(int(hour)).zfill(2))


def setPredict(data, max_dict, m, d, h):

    target_dir = 'lab/'
    GenerationModel = keras.models.load_model(target_dir + 'generation-0.002095-Tue May 17 14-35-13 2022.h5')
    ConsumptionModel = keras.models.load_model(target_dir + 'consumption-0.000363-Tue May 17 12-26-20 2022.h5')
    
    new_row = [None] * 5
    new_row[2], new_row[3], new_row[4] = m, d, h

    predictG_y = []
    predictC_y = []

    TEST_SIZE = 24
    for i in range(TEST_SIZE):

        tmpG = GenerationModel.predict(data)[0]
        tmpC = ConsumptionModel.predict(data)[0]

        tmpG_predict = normalizeRecover(tmpG[0], max_dict, 'generation')
        tmpC_predict = normalizeRecover(tmpC[0], max_dict, 'consumption')

        new_row[0], new_row[1] = tmpG[0], tmpC[0]
        new_row[2], new_row[3], new_row[4] = timeInc(new_row[2], new_row[3], new_row[4]) # Add one hour
        print(formatTime(new_row[2], new_row[3], new_row[4]) + ' done')

        new_row2D = np.array([new_row[0], new_row[1], new_row[2] / max_dict['month'], new_row[3] / max_dict['day'], new_row[4] / max_dict['hour']])
        data = np.array([np.vstack((data[0], new_row2D))[1:]])

        predictG_y.append(tmpG_predict)
        predictC_y.append(tmpC_predict)

    return predictG_y, predictC_y

def updateBuySellPrice(bid_res, buyP, sellP):

    last_month = str(bid_res.iloc[-1]['time']).split(' ')[0].split('-')[1]
    last_day = str(bid_res.iloc[-1]['time']).split(' ')[0].split('-')[2]
    mo = []
    da = []
    ho = []
    for data in bid_res['time']:
        mo.append(str(data).split(' ')[0].split('-')[1])
        da.append(str(data).split(' ')[0].split('-')[2])
        ho.append(str(data).split(' ')[1].split(':')[0])
    bid_res['month'] = mo
    bid_res['day'] = da
    bid_res['hour'] = ho
    bid_res = pd.DataFrame(bid_res)

    subset = bid_res[(bid_res['month'] == str(last_month)) & (bid_res['day'] == str(last_day))]

    bSubset = subset[subset['action'] == 'buy']
    if len(bSubset) > 0:
        bLast = bSubset['target_price'].iloc[-1]
        bMean = bSubset['trade_price'].mean()
        bMean = 1 if bMean < 1 else bMean
        bSuccessRatio = len(bSubset[bSubset['status'] != '未成交']) / len(bSubset)
        print('buy ratio = {}'.format(bSuccessRatio))
        print('buy last : {}'.format(bLast))

        buyP = bLast
        if bSuccessRatio < 0.0001 and bMean < 2.53:
            buyP = bMean if 2.51 < bMean else 2.51
        elif bSuccessRatio < 0.1 and buyP + 0.05 < 2.53:
            buyP = buyP + 0.05 
        elif bSuccessRatio < 0.3 and buyP + 0.02 < 2.53:
            buyP = buyP + 0.02
        elif bSuccessRatio < 0.5 and buyP + 0.01 < 2.53:
            buyP = buyP + 0.01
        elif bSuccessRatio > 0.95:
            buyP = buyP - 0.01
        
    sSubset = subset[subset['action'] == 'sell']
    if len(sSubset) > 0:
        sMin = 0.5
        sLast = sSubset['target_price'].iloc[-1]
        sMean = sSubset['trade_price'].mean()
        sMean = sMin if sMean < sMin else sMean
        sSuccessRatio = len(sSubset[sSubset['status'] != '未成交']) / len(sSubset)
        print('sell ratio = {}'.format(sSuccessRatio))
        print('sell last : {}'.format(sLast))

        sellP = sLast
        if sSuccessRatio < 0.1 and sellP - 0.05 > sMin:
            sellP = sellP - 0.05
        elif sSuccessRatio < 0.3 and sellP - 0.02 > sMin:
            sellP = sellP - 0.02
        elif sSuccessRatio < 0.5 and sellP - 0.01 > sMin:
            sellP = sellP - 0.01
        elif sSuccessRatio > 0.95:
            sellP = sellP + 0.01
    return np.round(buyP, 2), np.round(sellP, 2)


# You should not modify this part.
def config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()



if __name__ == "__main__":

    # constant dict of 50 files
    max_dict = {'generation': 5.240361907048226, 'consumption': 8.738419139009537, 'month': 12.0, 'day': 31.0, 'hour': 24.0}

    args = config()

    # get raw data, contains one day of generation, consumption info
    # split the 'time' column to 'month', 'day', 'hour'
    data_raw = setInputRawGC(args.generation, args.consumption, max_dict)

    # get the last time info from input data
    last_month = data_raw['month'].iloc[-1]
    last_day = data_raw['day'].iloc[-1]
    last_hour = data_raw['hour'].iloc[-1]
    print(formatTime(last_month, last_day, last_hour))
    
    # compress all the values to 0 ~ 1
    data_norm = normalize(data_raw, max_dict)

    # 2D to 3D, which is input size of LSTM model
    data_input = np.array([data_norm.values])

    # get the next 24 hours of generation and consumption data
    Gy, Cy = setPredict(data_input, max_dict, last_month, last_day, last_hour)

    # for output.csv
    bid_dict = { 'time': [], 'action': [], 'target_price': [], 'target_volume': []}

    # read the history bid-result info
    bid_res = pd.read_csv(args.bidresult)

    market_price = 2.53 # constant
    buyP, sellP = 2.51, 2.40 # start price (for the first day)
    if len(bid_res) > 0:
        buyP , sellP = updateBuySellPrice(bid_res, buyP, sellP) # update price by self-defined policy
    print('bp : ', buyP, 'sp : ', sellP)

    m, d, h = last_month, last_day, last_hour

    for i in range(len(Gy)):
        # ensure two prices are greater than 0
        Gy[i] = 0 if Gy[i] < 0 else Gy[i]
        Cy[i] = 0 if Cy[i] < 0 else Cy[i]
        balance = Gy[i] - Cy[i]

        # get the next hour time info
        m, d, h = timeInc(m, d, h)
        if(balance > 0):
            bid_dict['time'].append(formatTime(m, d, h))
            bid_dict['action'].append('sell')
            bid_dict['target_price'].append(sellP)
            bid_dict['target_volume'].append(balance)
        elif(balance < 0):
            bid_dict['time'].append(formatTime(m, d, h))
            bid_dict['action'].append('buy')
            bid_dict['target_price'].append(buyP)
            bid_dict['target_volume'].append(-balance)

        # if balance > 0.2 and h >= 7 and h <= 18: # 有餘
        #     if sellP > 2.53:
        #         bid_dict['time'].append(formatTime(m, d, h))
        #         bid_dict['action'].append('sell')
        #         bid_dict['target_price'].append(sellP)
        #         bid_dict['target_volume'].append(Gy[i]) # 賣的比市價高，那就全賣了
        #     else :
        #         bid_dict['time'].append(formatTime(m, d, h))
        #         bid_dict['action'].append('sell')
        #         bid_dict['target_price'].append(sellP)
        #         bid_dict['target_volume'].append(balance)
        # elif balance < -0.15: # 不足
        #     buyCount = balance * (buyP - market_price) # 在市場上"買 balance" 多花的錢
        #     sellCount = 1.0 * (sellP - market_price) # 在市場上"賣 1度電" 多賺的錢
        #     sumCount = np.max([buyCount, sellCount])
        #     buy = True if buyCount > sellCount else False

        #     if buy and sumCount > 0:
        #         bid_dict['time'].append(formatTime(m, d, h))
        #         bid_dict['action'].append('buy')
        #         bid_dict['target_price'].append(buyP)
        #         bid_dict['target_volume'].append(-balance)
        #     elif (not buy) and sumCount > 0:
        #         bid_dict['time'].append(formatTime(m, d, h))
        #         bid_dict['action'].append('sell')
        #         bid_dict['target_price'].append(sellP)
        #         bid_dict['target_volume'].append(Gy[i])

    bid_df = pd.DataFrame.from_dict(bid_dict)
    print(bid_df)

    plt.plot(Gy, 'go-', label='Generation')
    plt.plot(Cy, 'ro-', label='Consumption')
    plt.legend(loc = 'upper right')
    plt.xlabel('index')
    plt.ylabel('val')
    PREDICT_RESULT_PATH = 'predict/predict.png'
    pathlib.Path(PREDICT_RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PREDICT_RESULT_PATH)

    bid_df.to_csv(args.output, index=False)
    print('process complete')
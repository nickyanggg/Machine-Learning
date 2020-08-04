import sys
import pandas as pd
import numpy as np
data = pd.read_csv('./train.csv', encoding = 'big5')

remove_feature = ['AMB_TEMP', 'CH4', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
for rf in remove_feature:
    data = data[data.測項 != rf]

data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([8, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[8 * (20 * month + day) : 8 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 8 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:8*9 (9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][7, day * 24 + hour + 9] #value

mean_x = np.mean(x, axis = 0) #8 * 9 
std_x = np.std(x, axis = 0) #8 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #8 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

dim = 8 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 10
iter_time = 5000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_noise():
    # tmp = np.loadtxt("./data/cpuData.csv", dtype=np.float, delimiter=",")
    tmp = np.loadtxt("cpuData.csv", dtype=np.float, delimiter=",")
    x_train = tmp[0:, 0].astype(np.float)
    y_train = tmp[0:, 1].astype(np.float)
    noise = np.random.uniform(-0.2, 0.2, 720)
    x_list = x_train.tolist()
    y_list = y_train.tolist()
    noise_list = noise.tolist()

    for j in range(3):
        for i in range(720):
            x_list.append(720*(j+1)+1 + i)
            y_list.append(y_list[i] + noise_list[i])

    x_train = np.array(x_list)
    y_train = np.array(y_list)


def generate_10min():
    f = open('./data/cpuData_11_20.csv')
    df = pd.read_csv(f)
    time = np.array(df['time'])
    data = np.array(df['value'])
    data_list = data.tolist()
    train_y = []
    for i in range(1082):
        tmp = 0
        for j in range(10):
            if data_list[i*10+j] > tmp:
                tmp = data_list[i*10+j]
        train_y.append(tmp)
    train_y = np.array(train_y)[0:900]
    train_x = np.array(range(900))
    lable_y = np.array(train_y)[900:]
    lable_x = np.array(range(900, 1082))
    plt.figure()
    plt.plot(train_x, train_y, color='r')
    plt.show()
    return train_x, train_y


if __name__ == '__main__':
    lable_x = np.array(range(900, 1100))
    a = generate_10min()
    print(1)
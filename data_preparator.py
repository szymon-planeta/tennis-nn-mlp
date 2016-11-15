import random
import numpy as np
from logger import LOGGER


class DataPreparator(object):
    def __init__(self, all_data):
        self.all_data = all_data
        self.all_y = np.concatenate([np.ones([len(all_data),1]), np.zeros([len(all_data),1])], axis=1)

        
    def prepare_data(self):
        np.random.shuffle(self.all_data)
        for i in range(self.all_data.shape[0]):
            if random.random() < 0.5:
                tmp = np.copy(self.all_data[i][:5])
                self.all_data[i][:5] = self.all_data[i][6:] 
                self.all_data[i][6:] = tmp
                self.all_data[i][5] = 1 - self.all_data[i][5] if self.all_data[i][5] != -1 else self.all_data[i][5]
                self.all_y[i][0], self.all_y[i][1] = self.all_y[i][1], self.all_y[i][0]

        
    def get_datasets(self, N_train, N_test):
        self.prepare_data()
        train_X = self.all_data[:N_train]
        train_Y = self.all_y[:N_train]
        test_X = self.all_data[N_train:N_train+N_test]
        test_Y = self.all_y[N_train:N_train+N_test]
        return train_X, train_Y, test_X, test_Y
        
if __name__ == '__main__':
    a = np.arange(66).reshape((6, 11))
    d = DataPreparator(a)
    x_train, y_train, x_test, y_test = d.get_datasets(4, 2)
    print(x_train, y_train)
    print(x_test, y_test)
        
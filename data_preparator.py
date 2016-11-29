import random
import numpy as np
from logger import LOGGER
from pickler import serialize, deserialize


class DataPreparator(object):
    def __init__(self, all_data):
        self.prepared = False
        self.all_data = all_data
        self.all_y = np.concatenate([np.ones([len(all_data),1]), np.zeros([len(all_data),1])], axis=1)

    def standardize(self):
        for row in self.all_data:
            row[0] = 1.0 if row[0] == -1 else row[0]/900 #p1_rank
            row[1] = 0.4 if row[1] == -1 else row[1]        #p1_yr_winrate
            row[2] = 0.4 if row[2] == -1 else row[2]        #p1_2mon_winrate
            row[3] = 0.4 if row[3] == -1 else row[3]        #p1_court_winrate
            row[4] = 0.4 if row[4] == -1 else row[4]        #p1_tour_winrate            
            row[5] = 0.5 if row[5] == -1 else row[5]        #h2h
            row[6] = 1.0 if row[6] == -1 else row[6]/900 #p2_rank
            row[7] = 0.4 if row[7] == -1 else row[7]        #p2_yr_winrate
            row[8] = 0.4 if row[8] == -1 else row[8]        #p2_2mon_winrate
            row[9] = 0.4 if row[9] == -1 else row[9]        #p2_court_winrate
            row[10] = 0.4 if row[10] == -1 else row[10]  #p2_tour_winrate            


    def prepare_data(self):
        if not self.prepared:
            self.standardize()
            swapped = np.zeros(self.all_data.shape)
            for i in range(swapped.shape[0]):
                swapped[i][:5] = self.all_data[i][6:]
                swapped[i][5] = 1 - self.all_data[i][5]
                swapped[i][6:] = self.all_data[i][:5]

            new_y = np.concatenate([np.zeros([len(self.all_data),1]),
                                    np.ones([len(self.all_data), 1])], axis=1)
            self.all_data = np.concatenate([self.all_data, swapped])
            self.all_y = np.concatenate([self.all_y, new_y])
            x = list(zip(self.all_data, self.all_y))
            np.random.shuffle(x)
            self.all_data = np.asarray([a[0] for a in x]) 
            self.all_y = np.asarray([a[1] for a in x])     
        self.prepared = True
     
    def get_datasets(self, N_train, N_test):
        self.prepare_data()
        train_X = self.all_data[:N_train]
        train_Y = self.all_y[:N_train]
        test_X = self.all_data[N_train:N_train+N_test]
        test_Y = self.all_y[N_train:N_train+N_test]
        return train_X, train_Y, test_X, test_Y
    
    def save_datasets(self, N_train, N_test, name):
        self.prepare_data()
        train_X = self.all_data[:N_train]
        train_Y = self.all_y[:N_train]
        test_X = self.all_data[N_train:N_train+N_test]
        test_Y = self.all_y[N_train:N_train+N_test]
        
        serialize(train_X, name+'_train_X')
        serialize(train_Y, name+'_train_Y')
        serialize(test_X, name+'_test_X')
        serialize(test_Y, name+'_test_Y')
        
if __name__ == '__main__':
    a = np.arange(0, 0.66, 0.01).reshape((6, 11))
    d = DataPreparator(a)
    #x_train, y_train, x_test, y_test = d.save_datasets(4, 2, 'test')
    d.save_datasets(4, 2, 'test')
    x_train = deserialize('test_train_X')
    print(x_train)
    #print(x_train, y_train)
    #print(x_test, y_test)
        
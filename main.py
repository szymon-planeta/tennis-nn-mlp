from data_collector import DatabaseConnection, DataCollector
from data_preparator import DataPreparator
from mlp import MLP
from pickler import serialize, deserialize
from config import MDB, DRV, PWD
from logger import LOGGER


if __name__ == '__main__':
    #with DatabaseConnection(DRV, MDB, PWD) as con:
        #d = DataCollector(con)
        #matches = d.get_stats_for_last_n_matches(1000)
        #matches = d.get_stats_for_all_matches()

    file = 'all_matches'
    #serialize(matches, file)
    data = deserialize(file)
    #d = DataPreparator(data)
    #x_train, y_train, x_test, y_test = d.get_datasets(140000, 20000)
    
    #net = MLP([11, 10, 2], learn_coef=0.2, mom_coef=0.1)
    #net.start_training(x_train, y_train, x_test, y_test, epochs=600, tolerance=0.0001)    
    #serialize(net, 'net_file')
    
    net = deserialize('net_file')
    net.predict_single(data[140000])
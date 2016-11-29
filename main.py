from data_collector import DatabaseConnection, DataCollector
from data_preparator import DataPreparator
from mlp import MLP
from pickler import serialize, deserialize
from config import MDB, DRV, PWD
from logger import LOGGER
from datetime import datetime


if __name__ == '__main__':
    start = datetime.now()
    with DatabaseConnection(DRV, MDB, PWD) as con:
        d = DataCollector(con)
        #matches = d.get_stats_for_last_n_matches(10)
        matches = d.get_stats_for_all_matches()

    file = 'all_matches_13'
    serialize(matches, file)
    LOGGER.info("Serialized %s matches to file: %s", len(matches), file)
    end = datetime.now()
    LOGGER.info("Elapsed time: %s", end-start)
    #data = deserialize(file)
    #d = DataPreparator(data)
    
    #d.save_datasets(150000, 150000, fname)
    
    
    #fname = 'matches'
    #train_X = deserialize(fname+'_train_X')
    #train_Y = deserialize(fname+'_train_Y')
    #test_X = deserialize(fname+'_test_X')
    #test_Y = deserialize(fname+'_test_Y')
    
    #net = MLP([11, 10, 2], learn_coef=0.2, mom_coef=0.1)
    #net.start_training(train_X, train_Y, test_X, test_Y, epochs=5, tolerance=0.0001)    
    #serialize(net, 'net')
  
    # net=deserialize('net')
    # #net.test_multiple(test_X, test_Y)
    # with DatabaseConnection(DRV, MDB, PWD) as con:
        # d = DataCollector(con)
        # p1_id = d.get_player_id('Andy Murray')
        # #p1_id = d.get_player_id('Marin Cilic')
        # #p1_id = d.get_player_id('Kei Nishikori')
        # #p2_id = d.get_player_id('Stan Wawrinka')
        # #p2_id = d.get_player_id('Dominic Thiem')
        # #p1_id = d.get_player_id('Gael Monfils')
        # p2_id = d.get_player_id('Novak Djokovic')
        # #p1_id = d.get_player_id('David Goffin')
        # #p1_id = d.get_player_id('Milos Raonic')
        # tour_id = 13869
        # date = '2016-11-20'
        # match = d.get_stats_for_match(p1_id, p2_id, tour_id, date)
    # net.predict_single(match)

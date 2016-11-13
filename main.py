from data_collector import DatabaseConnection, DataCollector
from pickler import serialize, deserialize
from config import MDB, DRV, PWD
from logger import LOGGER

if __name__ == '__main__':
    with DatabaseConnection(DRV, MDB, PWD) as con:
        d = DataCollector(con)
        #matches = d.get_stats_for_last_n_matches(1000)
        matches = d.get_stats_for_all_matches()
    file = 'all_matches'
    serialize(matches, file)
    
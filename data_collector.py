# -*- coding: utf-8 -*-
import pyodbc
import datetime
from contextlib import contextmanager

from config import MDB, DRV, PWD
from logger import LOGGER


class DatabaseConnection(object):
    def __init__(self, driver, db_path, db_pw=None):
        self.driver = driver
        self.db_path = db_path
        self.db_pw = db_pw

    def __enter__(self):
        self.connection = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(self.driver, self.db_path, self.db_pw))
        return self.connection

    def __exit__(self, *args):
        self.connection.close()
      

      
class DataCollector(object):
    def __init__(self, connection):
    	self.con = connection
        
    @contextmanager
    def cursor(self):
        cur = self.con.cursor()
        yield cur
        cur.close()
        
    def execute_sql(self, SQL):
        with self.cursor() as cur:
            result = cur.execute(SQL).fetchall()
        return result
        
    def get_player_info(self, player_name, date):
        LOGGER.debug("Getting player: <%s> data for date: <%s>", player_name, date)
        p_name = player_name
        p_id = self.get_player_id(player_name)
        p_rank = self.get_player_ranking(p_id, date)
        LOGGER.debug("Player: <%s>  ID: <%s> Rank: <%s>", p_name, p_id, p_rank)
        
    def get_player_id(self, player_name):
        SQL = "SELECT ID_P FROM players_atp WHERE NAME_P='{}'".format(player_name)                                                                                                      
        result = self.execute_sql(SQL)
        p_id = result[0][0]
        return p_id

    def get_player_ranking(self, player_id, date):
        SQL = "SELECT TOP 1 POS_R FROM ratings_atp WHERE ID_P_R={} AND DATE_R<=#{}# ORDER BY DATE_R DESC;".format(player_id, date)
        result = self.execute_sql(SQL)
        p_rank = result[0][0]
        return p_rank


if __name__ == '__main__':
    p_name = 'Lukasz Kubot'
    date = "2013-10-13"
    with DatabaseConnection(DRV, MDB, PWD) as con:
        d = DataCollector(con)
        d.get_player_info(p_name, date)
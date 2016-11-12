# -*- coding: utf-8 -*-
import pyodbc
from contextlib import contextmanager
from datetime import datetime, timedelta

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
        self.d_format = '%Y-%m-%d'
        
    @contextmanager
    def cursor(self):
        cur = self.con.cursor()
        yield cur
        cur.close()
        
    def execute_sql(self, SQL):
        with self.cursor() as cur:
            result = cur.execute(SQL).fetchall()
        return result
        
    def get_stats_for_match(self, p1_id, p2_id, tour_id, date):
        p1_stats = self.get_player_info(p1_id, date, tour_id)
        p2_stats = self.get_player_info(p2_id, date, tour_id)
        h2h = self.get_head_to_head(p1_id, p2_id, date)
        stats = [*p1_stats, h2h, *p2_stats]
        LOGGER.debug("<%s> Tour ID: <%s> | Player 1 ID: <%s> | Player 2 ID: <%s> -- got stats vector:", date, tour_id, p1_id, p2_id)
        LOGGER.debug("<%s>", stats)
        
    def get_player_info(self, player, date, tour_id):
        p_id = player if isinstance(player, int) else self.get_player_id(player)
        LOGGER.debug("<%s> Getting player info: <%s>", date, player)
        p_rank = self.get_player_ranking(p_id, date)
        p_year_winrate = self.get_player_season_winrate(p_id, date)
        p_2mon_winrate = self.get_player_last_months_winrate(p_id, date, 2)
        court_type_id = self.get_tour_court_type_id(tour_id)
        p_court_winrate = self.get_player_court_winrate(p_id, court_type_id, date)
        p_tour_winrate = self.get_player_tour_winrate(p_id, tour_id, date)
        return [p_rank, p_year_winrate, p_2mon_winrate, p_court_winrate, p_tour_winrate]
        
    def get_player_id(self, player_name):
        SQL = "SELECT ID_P FROM players_atp WHERE NAME_P='{}'".format(player_name)                                                                                                      
        result = self.execute_sql(SQL)
        p_id = result[0][0]
        LOGGER.debug("Player name: %20s | Found ID: %s", player_name, p_id)
        return p_id

    def get_player_ranking(self, player_id, date):
        SQL = "SELECT TOP 1 POS_R FROM ratings_atp WHERE ID_P_R={} AND DATE_R<=#{}# ORDER BY DATE_R DESC;".format(player_id, date)
        result = self.execute_sql(SQL)
        p_rank = result[0][0]
        LOGGER.debug("<%s> Player ID: <%6s> | Rank: <%s>", date, player_id, p_rank)
        return p_rank
      
    def get_player_season_winrate(self, p_id, date):
        year = date[:4] + '-01-01'
        SQL = "SELECT  ID1_G, ID2_G FROM games_atp WHERE (ID1_G={id} OR ID2_G={id}) AND (DATE_G<#{date}# AND DATE_G>=#{year}#);".format(id=p_id, date=date, year=year)
        matches = self.execute_sql(SQL)
        wins = 0
        for match in matches:
            if match[0] == p_id:
                wins += 1
        winrate = wins/len(matches) 
        LOGGER.debug("<%s> Player ID: <%6s> | Yearly winrate: <%s>", date, p_id, winrate)
        return winrate
        
    def get_player_last_months_winrate(self, p_id, date, months):
        start_date = datetime.strptime(date, self.d_format)
        months_obj = timedelta(days=30*months)
        start_date -= months_obj
        start_date = start_date.strftime( self.d_format)
        SQL = "SELECT  ID1_G, ID2_G FROM games_atp WHERE (ID1_G={id} OR ID2_G={id}) AND (DATE_G<#{date}# AND DATE_G>=#{start_date}#);".format(id=p_id, date=date, start_date=start_date)
        matches = self.execute_sql(SQL)
        wins = 0
        for match in matches:
            if match[0] == p_id:
                wins += 1
        winrate = wins/len(matches) 
        LOGGER.debug("<%s> Player ID: <%6s> | Winrate since <%s>: <%s>", date, p_id, start_date, winrate)
        return winrate
        
    def get_tour_court_type_id(self, tour_id):
        SQL = "SELECT ID_C_T FROM tours_atp WHERE ID_T={};".format(tour_id)
        result = self.execute_sql(SQL)
        tour_court_type_id = result[0][0]
        LOGGER.debug("Tour ID: <%s> | Found court type ID: <%s>", tour_id, tour_court_type_id)
        return tour_court_type_id
       
    def get_player_court_winrate(self, p_id, court_type_id, date):
        SQL = ("SELECT  ID1_G, ID2_G FROM games_atp "
                    "INNER JOIN tours_atp ON games_atp.ID_T_G=tours_atp.ID_T "
                    "WHERE (ID1_G={id} OR ID2_G={id}) AND (DATE_G<#{date}# OR DATE_T<#{date}#)AND ID_C_T={c_t_id};").format(id=p_id, date=date, c_t_id=court_type_id)
        matches = self.execute_sql(SQL)
        wins = 0
        for match in matches:
              if match[0] == p_id:
                wins += 1
        winrate = wins/len(matches) 
        LOGGER.debug("<%s> Player ID: <%6s> | Winrate on court id <%s>: <%s>", date, p_id, court_type_id, winrate)
        return winrate
        
    def get_player_tour_winrate(self, p_id, tour_id, date):
        t_id = tour_id
        all_matches = []
        while t_id is not None:
            SQL = ("SELECT  ID1_G, ID2_G FROM games_atp "
                        "INNER JOIN tours_atp ON games_atp.ID_T_G=tours_atp.ID_T "
                        "WHERE (ID1_G={id} OR ID2_G={id}) AND (DATE_G<#{date}# OR DATE_T<#{date}#) "
                        "AND tours_atp.ID_T={tour_id};").format(id=p_id, date=date, tour_id=t_id)
            matches = self.execute_sql(SQL)
            all_matches += matches
            t_id = self.get_prev_tour(t_id)
        wins = 0
        for match in all_matches:
              if match[0] == p_id:
                wins += 1
        winrate = wins/len(all_matches) 
        LOGGER.debug("<%s> Player ID: <%6s> | Winrate in tour id <%s>: <%s>", date, p_id, tour_id, winrate)
        return winrate    
            
    def get_prev_tour(self, tour_id):
        SQL = "SELECT LINK_T FROM tours_atp WHERE ID_T={};".format(tour_id)
        try:
            prev_id = self.execute_sql(SQL)
            prev_id = prev_id[0][0]
        except:
            prev_id = None
        LOGGER.debug("Tour ID: <%s> | Got parent tour ID: <%s>", tour_id, prev_id)
        return prev_id
        
    def get_head_to_head(self, p1_id, p2_id, date):
        SQL = ("SELECT ID1_G, ID2_G FROM games_atp "
                    "INNER JOIN tours_atp ON games_atp.ID_T_G=tours_atp.ID_T "
                    "WHERE (ID1_G={p1_id} AND ID2_G={p2_id}) "
                    "OR (ID1_G={p2_id} AND ID2_G={p1_id}) "
                    "AND (DATE_G<#{date}# OR DATE_T<#{date}#) ;").format(p1_id=p1_id, p2_id=p2_id, date=date)
        matches = self.execute_sql(SQL)
        wins = 0
        for match in matches:
            if match[0] == p1_id:
                wins += 1
        p1_winrate = wins/len(matches)
        LOGGER.debug("Player 1 ID: <%6s> | Player 2 ID: <%6s> | Player 1 winrate: <%s>", p1_id, p2_id, p1_winrate)
        return p1_winrate
        
        
if __name__ == '__main__':
    with DatabaseConnection(DRV, MDB, PWD) as con:
        d = DataCollector(con)
        date = "2013-10-06"
        p1_name = 'Santiago Giraldo'
        p2_name = 'Lukasz Kubot'
        p1_id = d.get_player_id(p1_name)
        p2_id = d.get_player_id(p2_name)
        tour_id = 9862
        d.get_stats_for_match(p1_id, p2_id, tour_id, date)

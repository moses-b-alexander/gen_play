
import numpy as np


eps_tmp = 1e-12

s_char = "="
s_str = "\n" + (s_char * 68) + "\n"

screen_mode = "dark"

dtyp = np.float32

seed = 1

fps = 30

team_players = 11
all_players = 2 * team_players

ne_seasons, tb_seasons = [2017, 2018, 2019], [2020, 2021, 2022]
seasons = (
    [("NE", szn) for szn in ne_seasons] + [("TB", szn) for szn in tb_seasons]
)[::]

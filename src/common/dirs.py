
import os

from common.constants import screen_mode
from data.constants import match_count, season_count


root_dir = os.getcwd()

assets_dir = os.path.join(root_dir, "assets")
data_dir = os.path.join(root_dir, "amf_open_data")
logs_dir = os.path.join(root_dir, "logs")
output_dir = os.path.join(root_dir, "output")
processed_dir = \
    os.path.join(root_dir, f"processed_{season_count:02d}_{match_count:02d}")
if not os.path.exists(data_dir):  os.makedirs(data_dir)
if not os.path.exists(logs_dir):  os.makedirs(logs_dir)
if not os.path.exists(output_dir):  os.makedirs(output_dir)
if not os.path.exists(processed_dir):  os.makedirs(processed_dir)

statsbomb_logo_path = \
    os.path.join(assets_dir, f"HudlStatsbomb_Python_{screen_mode}.svg")

plays_path = os.path.join(data_dir, "plays")
tracking_path = os.path.join(data_dir, "tracking")

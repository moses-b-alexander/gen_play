
from common.constants import all_players, team_players


metadata_colnames = [
    "play_real", "play_padded",
    "play_is_first", "play_is_last",
    "game_playoffs",
]

reward_colname = "play_yards_gained"
# reward_colname = "play_xrec"

action_colnames = [
    [f"player_dx-{n:02d}", f"player_dy-{n:02d}"]
    for n in range(1, all_players + 1)
]
action_flattened_colnames =  [c for l in action_colnames for c in l]

play_state_colnames = [
    "game_season", "game_week",
    "play_defense_drive", "play_defense_game", "play_defense_in_lead",
    "play_defense_is_home_team", "play_defense_play", "play_defense_timeouts",
    "play_down", "play_formation_distance", "play_formation_entropy",
    "play_offense_drive", "play_offense_game", "play_offense_in_lead",
    "play_offense_is_home_team", "play_offense_play", "play_offense_timeouts",
    "play_quarter", "play_score_difference", "play_score_equal",
    "play_snap_center", "play_snap_left", "play_snap_right",
    "play_snap_x", "play_snap_y",
    "play_time",
    "play_yards_needed",
    "play_true_length",
]
player_state_colnames = [[
    f"player_ax-{n:02d}", f"player_ay-{n:02d}",
    f"player_position_db-{n:02d}", f"player_position_dl-{n:02d}",
    f"player_position_lb-{n:02d}",
    f"player_position_na-{n:02d}",
    f"player_position_ol-{n:02d}",
    f"player_position_qb-{n:02d}", f"player_position_rb-{n:02d}",
    f"player_position_te-{n:02d}", f"player_position_wr-{n:02d}",
    f"player_pre_snap-{n:02d}", f"player_post_snap-{n:02d}",
    f"player_vx-{n:02d}", f"player_vy-{n:02d}",
    f"player_x-{n:02d}", f"player_y-{n:02d}",
    f"player_yabs-{n:02d}",
    f"player_defense-{n:02d}", f"player_offense-{n:02d}",
] for n in range(1, all_players + 1)]
player_state_flattened_colnames = \
    [c for l in player_state_colnames for c in l]
state_colnames = play_state_colnames + player_state_flattened_colnames

num_pos_colnames = [
    "play_quarter", "play_time", "play_snap_x", "play_down",
    "play_defense_score", "play_offense_score",
    "play_defense_timeouts", "play_offense_timeouts",
    "play_season", "play_yards_needed",
]

play_remove_colnames = [
    "Unnamed: 0",
    "season", "gsis_play_id", "gsis_game_id", "gsis_old_game_id",
    "play_game_index", "play_drive_index",
    "play_start_event_index", "play_end_event_index",
    "offense_team_name", "defense_team_name",
    "play_offense_team_id", "play_defense_team_id",
    "play_defensive_back_depths", "play_linebacker_depths",
    "play_time_to_pass", "play_success", "play_points_won",
    "play_penalty_yards", "play_penalty_first_down",
    "play_penalty_down_loss", "play_penalty_offset",
    "play_return_yards", "play_field_goal_result", "play_extra_point_result",
    "play_kick_yards", "play_kick_hangtime", "play_kick_blocked",
    "play_kick_fair_catch", "play_kick_downed", "play_points_lost",
    "play_start_position_yards", "play_explosive", "play_box_players",
    "play_offensive_personnel", "play_defensive_personnel",
    "play_offense_penalty_accepted", "play_defense_penalty_accepted",
    "play_offense_flagged", "play_defense_flagged", "play_epoch_uuid",
    "play_offense_players_left", "play_offense_players_right",
    "play_point_of_attack_y", "play_point_of_attack_dy",
    "play_pocket_front_x", "play_pocket_back_x",
    "play_pocket_left_y", "play_pocket_right_y",
    "play_qb_exit_pocket_x", "play_qb_exit_pocket_y",
    "play_qb_exit_pocket_clock", "play_run_gap", "play_qb_dropback_depth",
    "play_run_behind_1", "play_run_behind_2", "play_run_at",
    "play_derived_pressure_gaps", "play_pass_air_yards_to_the_sticks",
    "play_pass_within_pocket", "play_pass_comp_prob",
    "play_start_position", "play_pass_placement_displacement",
    "play_target_separation", "play_catch_separation",
    "play_formation_into_boundary", "play_drop_eight",
    "play_catch_prob_with_placement", "play_catch_prob_without_placement",
    "play_initial_formation",
    "play_initial_formation_run_strength",
    "play_initial_formation_pass_strength",
    "play_initial_formation_snap_side", "play_initial_formation_fib",
    "play_formation_run_strength", "play_formation_pass_strength",
    "play_formation_fib", "play_formation",
    "play_punt_snap_time", "play_punt_operation_time", "play_punt_hang_time",
    "play_net_punt_yards", "play_gross_punt_yards", "play_punt_return_yards",
    "play_punt_roll_yards", "play_punt_air_yards",
    "play_punt_block_point_x", "play_punt_block_point_y", "play_punt_outcome",
    "play_kickoff_x", "play_kickoff_y",
    "play_kickoff_land_x", "play_kickoff_land_y", "play_kickoff_outcome",
    "play_kick_hang_time", "play_kickoff_return_yards",
    "play_kickoff_return_chunk", "play_kickoff_return_explosive",
    "play_fg_snap_time", "play_fg_operation_time", "play_fg_attempted",
    "play_fg_made", "play_fg_kick_blocked", "play_fg_length",
    "play_xp_snap_time", "play_xp_operation_time", "play_xp_attempted",
    "play_xp_made", "play_xp_kick_blocked",
    "play_qb_pressure", "play_qb_hit", "play_qb_sacked",
    "play_qb_scramble", "play_qb_rush",
    "play_fumble_forced", "play_fumble_lost", "play_fumble_out_of_bounds",
    "play_tackle_types", "play_pass_location",
    "play_clock_paused", "play_pre_snap_motion", "play_snap_hurried",
    "play_action_pass", "play_receiver_screen_pass", "play_designed_pass",
    "play_blitz", "play_simulated_pressure", "play_contested_catch_attempt",
    "play_first_down_won", "play_touchdown_won",
    "play_shotgun", "play_pass_outcome", "play_yards_after_contact",
    "play_tackle_success", "play_handoff_received",
    "play_included_fake", "play_motion_type", "play_havoc",
    "play_prob_field_goal", "play_prob_no_score", "play_prob_opp_field_goal",
    "play_prob_opp_safety", "play_prob_opp_touchdown", "play_prob_safety",
    "play_prob_touchdown", "play_ep", "play_scrimmage_epa",
    "play_prob_made_field_goal", "play_fg_ep", "play_fg_epa",
    "play_snap_locations",
    "play_player_role_x", "play_player_role_y", "play_player_role_z",
    "play_player_role_f", "play_player_role_h",
    "play_start_video_timestamp_angle2", "play_end_video_timestamp_angle2",
    "play_start_video_timestamp_angle3", "play_end_video_timestamp_angle3",
    "play_video_info", "play_two_minute", "play_four_minute",
    "play_defensive_coverage", "play_blitz_type", "play_blitz_players",
    "play_blitz_jersey_numbers", "play_blitz_depth", "play_pass_rushers",
    "play_pass_coverage_players", "play_defensive_front",
    "play_defensive_front_players", "play_defensive_front_num_players",
    "play_mofo", "play_zone_coverage_prob", "play_offensive_substitutions",
    "play_defensive_substitutions", "play_incompletion_types",
    "play_garbage_time", "play_offensive_players",
    "play_offensive_skill_players", "play_defensive_players",
    "play_pass_direction", "play_pass_width", "play_bunched_players",
    "play_previous_play_uuid", "play_next_play_uuid", "play_run_gap_1yard",
    "play_yards_run_from_los", "play_target_alignment_position_id",
    "play_target_route", "gsis_play_id_db", "gsis_play_id_fix", "null_gsis",
    "play_yards_net",
]
track_remove_colnames = [
    "play_offense_team_id", "team_id", "nfl_team_id", "ngs_x", "ngs_y",
    "offense_left_to_right", "position_code", "game_id",
]

frame_colnames = ["play_uuid", "frame_time", "frame_time_snap"]

player_bool_flattened_colnames = [[
    f"player_defense-{n:02d}", f"player_offense-{n:02d}",
    f"player_position_na-{n:02d}",
    f"player_position_qb-{n:02d}", f"player_position_rb-{n:02d}",
    f"player_position_wr-{n:02d}", f"player_position_te-{n:02d}",
    f"player_position_ol-{n:02d}", f"player_position_dl-{n:02d}",
    f"player_position_db-{n:02d}", f"player_position_lb-{n:02d}",
    f"player_pre_snap-{n:02d}", f"player_post_snap-{n:02d}",
] for n in range(1, all_players + 1)]
player_bool_flattened_colnames = \
    [cc for c in player_bool_flattened_colnames for cc in c]

player_float_flattened_colnames = [[
    f"player_x-{n:02d}", f"player_y-{n:02d}",
    f"player_dx-{n:02d}", f"player_dy-{n:02d}",
    f"player_vx-{n:02d}", f"player_vy-{n:02d}",
    f"player_ax-{n:02d}", f"player_ay-{n:02d}",
    f"player_yabs-{n:02d}",
] for n in range(1, all_players + 1)]
player_float_flattened_colnames = \
    [cc for c in player_float_flattened_colnames for cc in c]

player_delta_colnames = [
    [f"player_dx-{n:02d}", f"player_dy-{n:02d}"]
    for n in range(1, all_players + 1)
]
player_delta_flattened_colnames = \
    [c for l in player_delta_colnames for c in l]

player_location_colnames = [
    [f"player_x-{n:02d}", f"player_y-{n:02d}"]
    for n in range(1, all_players + 1)
]
player_location_flattened_colnames = \
    [c for l in player_location_colnames for c in l]

player_position_flattened_colnames = [
    c for c in player_bool_flattened_colnames if "player_position_" in c]

play_xrec_derived_colnames = [
    "game_season", "play_down",
    "play_quarter", "play_score_difference",
    "play_snap_center", "play_snap_left", "play_snap_right",
    "play_snap_x", "play_yards_needed",
]

player_uuid_colnames = [
    f"player_uuid-{n:02d}" for n in range(1, all_players + 1)
]

import numpy as np
import pandas as pd
from .glicko import calculate_win_prob
from datetime import timedelta


def predict_win_probabilities(players, opponents, periods, ratings_history):

    preds = list()

    for cur_player, cur_opponent, cur_period in zip(players, opponents, periods):

        player_mean, player_var = ratings_history[cur_period][cur_player]
        opponent_mean, opponent_var = ratings_history[cur_period][cur_opponent]

        preds.append(
            calculate_win_prob(player_mean, opponent_mean, player_var, opponent_var)
        )

    preds = np.array(preds)

    return preds


def fetch_player_history(player_name, period_length_days, start_date, rating_history):

    min_period = 0
    max_period = len(rating_history)

    period_dates = [
        start_date + timedelta(days=i * period_length_days)
        for i in range(min_period, max_period)
    ]

    history = pd.DataFrame([x[player_name] for x in rating_history])
    history.columns = ["mean", "variance"]
    history.index = period_dates

    return history

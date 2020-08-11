import numpy as np
import pandas as pd
from .glicko import calculate_win_prob
from datetime import timedelta


def predict_win_probabilities(players, opponents, periods, ratings_history):
    """Predicts the win probabilities for players given Glicko ratings.

    Args:
    players: A numpy array of player names.
    opponents: A numpy array of opponent names.
    periods: The periods the contest were played in.
    ratings_history: The list of ratings returned by running
        glicko.calculate_ratings.
    
    Returns:
    An array of win probabilities with one entry for each of the specified
    matches.
    """

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
    """Extracts the history for a single player from the ratings history.

    Args:
    player_name: The player whose history to fetch.
    period_length_days: The period length used to fit Glicko.
    start_date: The date corresponding to period = 0.
    rating_history: The ratings history returned by glicko.calculate_ratings.

    Returns:
    A pandas DataFrame whose index is the date and whose rows give the mean and
    variance of the rating for each date.
    """

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

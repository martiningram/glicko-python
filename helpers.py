from glicko import calculate_win_prob


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

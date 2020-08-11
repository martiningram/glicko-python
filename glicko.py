import numpy as np
from collections import defaultdict
from scipy.optimize import minimize


q = np.log(10) / 400.0


def g(sigma_sq):
    """Calculates the quantity g as given in the Glicko paper."""

    return 1.0 / np.sqrt(1 + 3 * q ** 2 * sigma_sq / np.pi ** 2)


def E(mu, mu_j, sigma_j_sq):
    """Calculates the quantity E as given in the Glicko paper."""

    return 1.0 / (1 + 10 ** (-g(sigma_j_sq) * (mu - mu_j) / 400))


def delta_sq(mu, mu_j, sigma_j_sq, n_j):
    """Calculates the quantity delta squared as given in the Glicko paper."""

    e_term = E(mu, mu_j, sigma_j_sq)

    to_sum = n_j * g(sigma_j_sq) ** 2 * e_term * (1 - e_term)

    return (q ** 2 * np.sum(to_sum)) ** (-1)


def calculate_new_rating(mu, sigma_sq, mu_j, sigma_j_sq, n_j, s_jk):
    """Calculates the updated mean and variance for the player.

    mu: Prior mean. A single number.
    sigma_sq: Prior variance. A single number.
    mu_j: The means of the m opponents faced in the rating period -- an array of
        shape [m,].
    sigma_j_sq: The variances of the m opponents faced in the rating period --
        an array of shape [m,].
    n_j: The number of contests played against each of the opponents -- an array
        of shape [m,].
    s_jk: The outcomes of the contests against each of the m players. This is a
        ragged array, represented as a list. The list has m elements; element j
        of the list has n_j entries, one for each of the contests. Each entry is
        1 if the player won, and zero otherwise.
    
    Returns: A Tuple of the new mean and variance.
    """

    delta_squared = delta_sq(mu, mu_j, sigma_j_sq, n_j)

    sum_result = 0.0
    total_discrepancy = 0.0

    for cur_sjk, cur_mu_j, cur_sigma_j_sq, cur_n_j in zip(s_jk, mu_j, sigma_j_sq, n_j):

        sum_result += np.sum(
            cur_n_j * g(cur_sigma_j_sq) * (cur_sjk - E(mu, cur_mu_j, cur_sigma_j_sq))
        )

        total_discrepancy += np.sum(
            calculate_discrepancy(mu, cur_mu_j, sigma_sq, cur_sigma_j_sq, cur_sjk)
        )

    mean_pre_factor = q / ((1 / sigma_sq + 1 / delta_squared))

    new_mean = mu + mean_pre_factor * sum_result

    new_variance = (1 / sigma_sq + 1 / delta_squared) ** (-1)

    return new_mean, new_variance, total_discrepancy


def update_time_passage(ratings, period_to_period_variance):
    """Adds the variance from the passage of time."""

    for cur_player in ratings:
        ratings[cur_player] = (
            ratings[cur_player][0],
            ratings[cur_player][1] + period_to_period_variance,
        )

    return ratings


def calculate_win_prob(mu_i, mu_j, sigma_i_sq, sigma_j_sq):
    """Calculates the win probability p_ij as given in equation 16 of the Glicko paper."""

    exponent = -g(sigma_i_sq + sigma_j_sq) * (mu_i - mu_j) / 400.0

    return 1.0 / (1 + 10 ** (exponent))


def calculate_discrepancy(mu_i, mu_j, sigma_i_sq, sigma_j_sq, s_ij):
    """Calculates the discrepancy between the predicted and actual outcome, as
    given in equation 15 of the Glicko paper."""

    p_ij = calculate_win_prob(mu_i, mu_j, sigma_i_sq, sigma_j_sq)

    return -s_ij * np.log(p_ij) - (1 - s_ij) * np.log(1 - p_ij)


def calculate_ratings(
    winners, losers, periods, prior_variance, period_to_period_variance
):
    """Calculates Glicko ratings.

    Args:
    winners: The array of winners for each of N contests, of shape [N,].
    losers: The array of losers for each of N contests, of shape [N,].
    periods: The period each match was played in. This must be an integer
        value. Also of shape [N,].
    prior_variance: The initial variance for each player.
    period_to_period_variance: The variance to add between time periods.

    Returns:
    A Tuple of two elements. 
    The first is a list of length T + 1, where T is the maximum period. Each
    entry of this list contains a dictionary with the ratings for each
    competitor _prior_ to the update in this period. This is convenient since
    each entry can be directly used for prediction -- e.g. to predict period 3,
    the most up-to-date ratings will be in entry 3 of the list [assuming periods
    start with 0].
    The second is the total discrepancy, a measure of the fit of the ratings.
    """

    start_period = np.min(periods)
    end_period = np.max(periods)

    prior_ratings = defaultdict(lambda: (1500.0, prior_variance))
    rating_history = list()

    total_discrepancy = 0.0

    for cur_period in np.arange(start_period, end_period + 1):

        if cur_period != start_period:
            # Update from passage of time
            prior_ratings = update_time_passage(
                prior_ratings, period_to_period_variance
            )

        rating_history.append(prior_ratings.copy())

        new_ratings = prior_ratings.copy()

        cur_winners = winners[periods == cur_period]
        cur_losers = losers[periods == cur_period]

        unique_players = np.union1d(cur_winners, cur_losers)

        for cur_unique_player in unique_players:

            # Summarise the outcomes as required
            was_winner = cur_winners == cur_unique_player
            was_loser = cur_losers == cur_unique_player

            # Find opponents
            relevant_winners = cur_winners[was_winner | was_loser]
            relevant_losers = cur_losers[was_winner | was_loser]

            opponents = np.select(
                [
                    relevant_winners == cur_unique_player,
                    relevant_losers == cur_unique_player,
                ],
                [relevant_losers, relevant_winners],
            )

            # This is s, whether or not the match was a win
            outcomes = opponents == relevant_losers

            # We now need to aggregate s by opponent
            s_jk = list()

            unique_opponents = np.unique(opponents)

            for cur_opponent in unique_opponents:

                cur_outcomes = outcomes[opponents == cur_opponent].astype(int)
                s_jk.append(cur_outcomes)

            n_j = np.array([len(x) for x in s_jk])

            opponent_means = np.array(
                [prior_ratings[cur_opponent][0] for cur_opponent in unique_opponents]
            )

            opponent_variances = np.array(
                [prior_ratings[cur_opponent][1] for cur_opponent in unique_opponents]
            )

            # We can now update the player
            player_mu, player_sigma_sq = prior_ratings[cur_unique_player]

            new_mu, new_sigma_sq, discrepancy = calculate_new_rating(
                player_mu,
                player_sigma_sq,
                opponent_means,
                opponent_variances,
                n_j,
                s_jk,
            )

            total_discrepancy += discrepancy

            new_ratings[cur_unique_player] = (new_mu, new_sigma_sq)

        prior_ratings = new_ratings

    rating_history.append(update_time_passage(prior_ratings, period_to_period_variance))

    return rating_history, total_discrepancy


def find_optimal_parameters(
    winners,
    losers,
    periods,
    start_prior_variance=100 ** 2,
    start_period_to_period_variance=10 ** 2,
    verbose=True,
    tolerance=1e-3,
):
    def fun_to_minimize(theta):

        # Constrain
        prior_variance, period_to_period_variance = np.exp(theta)

        _, discrepancy = calculate_ratings(
            winners, losers, periods, prior_variance, period_to_period_variance
        )

        if verbose:
            print(
                f"Prior variance: {prior_variance:.2f}; "
                f"period-to-period variance: {period_to_period_variance:.2f}; "
                f"discrepancy: {discrepancy:.2f}"
            )

        return discrepancy

    opt_result = minimize(
        fun_to_minimize,
        np.log([start_prior_variance, start_period_to_period_variance]),
        tol=tolerance,
    )

    fit_x = np.exp(opt_result.x)

    return (
        opt_result.success,
        {"prior_variance": fit_x[0], "period_to_period_variance": fit_x[1]},
    )

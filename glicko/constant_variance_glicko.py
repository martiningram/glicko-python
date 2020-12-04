import numpy as np
from .glicko import calculate_new_rating, calculate_win_prob
from collections import defaultdict
from scipy.optimize import minimize


def calculate_ratings(winners, losers, variance):

    ratings = defaultdict(lambda: 1500.0)

    total_discrepancy = 0.0

    history = list()

    for cur_winner, cur_loser in zip(winners, losers):

        winner_rating, loser_rating = ratings[cur_winner], ratings[cur_loser]

        # Create the required data structures
        mu_winner = np.array([winner_rating])
        mu_loser = np.array([loser_rating])
        variance_array = np.array([variance])
        n = np.array([1])

        new_mean_winner, _, discrepancy = calculate_new_rating(
            mu_winner, variance_array, mu_loser, variance_array, n, [np.array([1])]
        )

        # No need for discrepancy here since we would be double-counting the outcome
        new_mean_loser, _, _ = calculate_new_rating(
            mu_loser, variance_array, mu_winner, variance_array, n, [np.array([0])]
        )

        total_discrepancy += discrepancy

        ratings[cur_winner] = new_mean_winner
        ratings[cur_loser] = new_mean_loser

        history.append(
            {
                "prior_mu_winner": winner_rating,
                "prior_mu_loser": loser_rating,
                "prior_win_prob": calculate_win_prob(
                    winner_rating, loser_rating, variance, variance
                ),
            }
        )

    return ratings, history, total_discrepancy


def find_optimal_parameters(
    winners, losers, start_variance=100 ** 2, verbose=True, tolerance=1e-2,
):
    """Fits the parameters in constant-variance Glicko.

    Args:
    winners: The array of winners for each of N contests, of shape [N,].
    losers: The array of losers for each of N contests, of shape [N,].
    start_variance: The initialisation for the fixed variance for each player in
        the optimisation.
    verbose: Whether or not to print the progress of the optimisation.
    tolerance: The tolerance required for the optimisation to successfully
        terminate.
    
    Returns:
    A Tuple whose first entry is a flag indicating the optimisation's success,
    and the second is a dictionary containing one element: the optimal prior
    variance.
    """

    def fun_to_minimize(theta):

        # Constrain
        variance = theta[0] ** 2

        _, _, discrepancy = calculate_ratings(winners, losers, variance)

        if verbose:
            print(
                f"Player standard deviation: {np.sqrt(variance):.2f}; "
                f"discrepancy: {discrepancy:.2f}"
            )

        return discrepancy

    opt_result = minimize(fun_to_minimize, np.sqrt([start_variance]), tol=tolerance,)

    fit_x = opt_result.x ** 2

    return (opt_result.success, {"variance": fit_x[0]})

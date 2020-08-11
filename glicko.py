import numpy as np


q = np.log(10) / 400.0


def g(sigma_sq):

    return 1.0 / np.sqrt(1 + 3 * q ** 2 * sigma_sq / np.pi ** 2)


def E(mu, mu_j, sigma_j_sq):

    return 1.0 / (1 + 10 ** (-g(sigma_j_sq) * (mu - mu_j) / 400))


def delta_sq(mu, mu_j, sigma_j_sq, n_j):

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
        of the list has n_j entries, one for each of the contests. The entry is
        1 if the player won, and zero otherwise.
    
    Returns: A Tuple of the new mean and variance.
    """

    delta_squared = delta_sq(mu, mu_j, sigma_j, n_j)

    sum_result = 0.0

    for cur_sjk, cur_mu_j, cur_sigma_j_sq, cur_n_j in zip(s_jk, mu_j, sigma_j_sq, n_j):

        sum_result += np.sum(
            cur_n_j * g(cur_sigma_j_sq) * (cur_sjk - E(mu, cur_mu_j, cur_sigma_j_sq))
        )

    mean_pre_factor = q / ((1 / sigma_sq + 1 / delta_squared))

    new_mean = mu + mean_pre_factor * sum_result

    new_variance = (1 / sigma_sq + 1 / delta_squared) ** (-1)

    return new_mean, new_variance

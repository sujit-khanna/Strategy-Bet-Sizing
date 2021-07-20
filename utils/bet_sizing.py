
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


def discrete_signal(signal0, step_size):

    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor
    return signal1


def prob_to_bet_size(prob, num_classes, pred=None, step_size=None):
    """
    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this
    representation, the probability will always be between 1/num_classes and 1.0.
    """
    #
    if prob.shape[0] == 0:
        return pd.Series(dtype='float64')

    bet_sizes = (prob - 1/num_classes) / (prob * (1 - prob))**0.5

    if not isinstance(pred, type(None)):
        bet_sizes = pred * (2 * norm.cdf(bet_sizes) - 1)
    else:
        bet_sizes = bet_sizes.apply(lambda s: 2 * norm.cdf(s) - 1)

    if step_size is not None:
        bet_sizes = discrete_signal(bet_sizes, step_size)

    return bet_sizes


def kelly_betting(win_probability: float, profit_unit: float, loss_unit: float) -> Tuple[float, float]:
    """
    Calculates optimal bet size based on the kelly criterion.
    """
    loss_probability = 1 - win_probability

    kelly_bet_size = win_probability / loss_unit - loss_probability / profit_unit
    expected_growth_rate = loss_probability * np.log(1 - loss_unit * kelly_bet_size) + \
                           win_probability * np.log(1 + profit_unit * kelly_bet_size)

    return kelly_bet_size, expected_growth_rate

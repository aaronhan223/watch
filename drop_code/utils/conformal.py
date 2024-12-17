import numpy as np
from sklearn.metrics import pairwise_distances
from operator import mul
from scipy.special import gammainc, gamma, gammaln
from functools import reduce


def knn_conformal_measure(X, y):
    """
    Classic nearest neighbors distance score
    """

    if y.size == 1:
        return [1]

    distances = pairwise_distances(X)

    np.fill_diagonal(distances, np.inf)

    scores = list()

    for cur_index, cur_label in enumerate(y):
        dist_same_class = distances[cur_index][y == cur_label]
        dist_other_class = distances[cur_index][y != cur_label]

        if dist_same_class.size == 0 and dist_other_class.size == 0:
            scores += [[1]]
        elif dist_same_class.size == 0 and dist_other_class.size != 0:
            # think more about this case
            scores += [[1]]
        elif dist_same_class.size != 0 and dist_other_class.size == 0:
            scores += [[0]]
        else:
            closest_same_cass = min(dist_same_class)
            closest_other_cass = min(dist_other_class)
            scores += [closest_other_cass/closest_same_cass]

    return scores


def label_conformity_measure(scores, labels, num_classes=2):
    unique_labels = np.unique(labels)
    label_scores = np.zeros(num_classes)
    for cur_label in unique_labels:
        cur_scores = [cur_sc for cur_index, cur_sc in enumerate(
            scores) if labels[cur_index] == cur_label]
        label_scores[cur_label] = np.mean(cur_scores)

    final_scores = [label_scores[cur_label] for _, cur_label in enumerate(
        labels)]

    return final_scores


def label_conformal_transducer(scores, randomized=True):
    """
    scores are ordered
    """
    u = np.random.uniform()
    if len(scores) == 1:
        if randomized:
            p_val = u
        else:
            p_val = 1
        return p_val

    last_score = scores[-1]
    num_of_scores = len(scores)
    smaller_scores = 0
    equal_scores = 0

    for cur_score in scores:
        if cur_score < last_score:
            smaller_scores += 1
        if cur_score == last_score:
            equal_scores += 1

    if randomized:
        p_val = (smaller_scores+u*equal_scores)/num_of_scores
    else:
        p_val = (smaller_scores+equal_scores)/num_of_scores

    return p_val


def label_conditional_conformal_transducer(scores, labels, randomized=True):
    """
    scores are ordered
    """
    u = np.random.uniform()
    if len(scores) == 1:
        if randomized:
            p_val = u
        else:
            p_val = 1
        return p_val

    last_score = scores[-1]

    # select scores corresponding to the target class
    target_label = labels[-1]

    target_scores = [cur_score for cur_index, cur_score in enumerate(
        scores) if labels[cur_index] == target_label]

    num_of_scores = len(target_scores)
    smaller_scores = 0
    equal_scores = 0

    for cur_score in target_scores:
        if cur_score < last_score:
            smaller_scores += 1
        if cur_score == last_score:
            equal_scores += 1

    if randomized:
        p_val = (smaller_scores+u*equal_scores)/num_of_scores
    else:
        p_val = (smaller_scores+equal_scores)/num_of_scores

    return p_val


def simple_bet(epsilon, p):
    assert epsilon < 1
    assert epsilon > 0
    return epsilon * (p ** (epsilon-1))


# def next_val_martingale_simple_bets(old_martingale_value, new_p_val, epsilon):

#     return old_martingale_value*simple_bet(epsilon, new_p_val)


def martingale_simple_bets(p_vals, epsilon=0.5):
    if len(p_vals) == 0:
        # on a logarithmic scale
        return 0
    mart_value = list()
    mart_value += [0]
    for cur_p_val in p_vals:
        mart_value += [mart_value[-1]+np.log(simple_bet(epsilon, cur_p_val))]
    return mart_value[1:]


def simple_mixture(p_vals):
    if len(p_vals) == 0:
        return 1

    b = - sum(np.log(p_vals))

    # np.log(reduce(mul, p_vals))
    n = len(p_vals)

    log_norm_inc_gamma_fn = np.log(gammainc(n+1, b))
    log_gamma_term = gammaln(n+1)

    log_smpl_mixt = b - (n+1)*np.log(b) + \
        log_gamma_term + log_norm_inc_gamma_fn
    return log_smpl_mixt

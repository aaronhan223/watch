import numpy as np
from .concentrations import betting_ci_upper_limit

class Set_valued_predictor_wrapper(object):
    """
    Wrapper that transforms a point predictor into a set-valued one
    """
    def __init__(self):
        self.base_predictor = None
        self.lmbd_star = None
        self.n_lmbds = 100

        self.alpha = None
        self.delta = None
        self.search_tol = 1e-3

    def fit(self, X_cal, y_cal):

        candidate_lmbds = np.linspace(0, 1, self.n_lmbds)[::-1]

        #do binary search

        cand_lmbd_left = 0
        cand_lmbd_right = 1
        cand_lmbd_mid = 0.5

        cur_sets_left = self.predict_sets(X_cal, cand_lmbd=cand_lmbd_left)
        cur_sets_right = self.predict_sets(X_cal, cand_lmbd=cand_lmbd_right)
        cur_sets_mid = self.predict_sets(X_cal, cand_lmbd=cand_lmbd_mid)

        num_of_preds = len(cur_sets_left)

        misclas_losses_left = [
            y_cal[i] not in cur_sets_left[i] for i in range(num_of_preds)
        ]
        misclas_losses_right = [
            y_cal[i] not in cur_sets_right[i] for i in range(num_of_preds)
        ]
        misclas_losses_mid = [
            y_cal[i] not in cur_sets_mid[i] for i in range(num_of_preds)
        ]

        risk_ucb_left = betting_ci_upper_limit(misclas_losses_left, self.delta)
        risk_ucb_right = betting_ci_upper_limit(misclas_losses_right,
                                                self.delta)
        risk_ucb_mid = betting_ci_upper_limit(misclas_losses_mid, self.delta)

        while abs(cand_lmbd_left - cand_lmbd_right) > self.search_tol:
            if risk_ucb_mid >= self.alpha:
                cand_lmbd_left = cand_lmbd_mid
                cand_lmbd_mid = (cand_lmbd_left + cand_lmbd_right) / 2
                cur_sets_mid = self.predict_sets(X_cal,
                                                 cand_lmbd=cand_lmbd_mid)
                misclas_losses_mid = [
                    y_cal[i] not in cur_sets_mid[i]
                    for i in range(num_of_preds)
                ]
                risk_ucb_mid = betting_ci_upper_limit(misclas_losses_mid, self.delta)
            else:
                cand_lmbd_right = cand_lmbd_mid
                cand_lmbd_mid = (cand_lmbd_left + cand_lmbd_right) / 2
                cur_sets_mid = self.predict_sets(X_cal,
                                                 cand_lmbd=cand_lmbd_mid)
                misclas_losses_mid = [
                    y_cal[i] not in cur_sets_mid[i]
                    for i in range(num_of_preds)
                ]
                risk_ucb_mid = betting_ci_upper_limit(misclas_losses_mid, self.delta)

        if risk_ucb_mid >= self.alpha:
            self.lmbd_star = cand_lmbd_right
        else:
            self.lmbd_star = cand_lmbd_mid
            
    def predict_sets(self, X_test, cand_lmbd=None):

        # predict probabilities
        probs = self.base_predictor.predict(X_test)

        num_of_preds, _ = probs.shape

        # sort predicted probabilities for each point in decreasing order
        prob_sort = -np.sort(-probs, axis=1)

        # sort predicted classes for each point in decreasing order from most likely
        classes_sort = np.argsort(-probs, axis=1)

        # get cumulative probs of most likely classes
        cumulative_probs = prob_sort.cumsum(axis=1)

        # get cumulative probs of exceeding (more likely) classes
        more_likely_probs = cumulative_probs - prob_sort

        if cand_lmbd is None:
            sets = [
                np.sort(classes_sort[cur_point][
                    more_likely_probs[cur_point] <= self.lmbd_star])
                for cur_point in range(num_of_preds)
            ]
        else:
            #if this function is called during training
            sets = [
                np.sort(classes_sort[cur_point][
                    more_likely_probs[cur_point] <= cand_lmbd])
                for cur_point in range(num_of_preds)
            ]

        return sets

    def eval_pred(self, X, y):
        pred_sets = self.predict_sets(X)

        num_of_preds = len(pred_sets)

        # evaluate coverage
        misclas_risk = np.mean(
            [y[i] not in pred_sets[i] for i in range(num_of_preds)])
        # evaluate size
        length = np.mean([len(cur_set) for cur_set in pred_sets])

        return misclas_risk, length
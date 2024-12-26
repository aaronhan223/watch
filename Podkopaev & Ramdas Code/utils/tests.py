import numpy as np
from .concentrations import hoeffding_ci_upper_limit, pm_bernstein_ci_upper_limit, betting_ci_upper_limit
from .concentrations import pm_hoeffding_lower_limit, pm_bernstein_lower_limit, betting_cs_lower_limit
from .concentrations import conjmix_empbern_cs, conjmix_hoeffding_cs
from sklearn.preprocessing import OneHotEncoder


def misclas_losses(x, y): return x != y


def brier_scores(y_true, pred_probs, n_classes=2):
    """
    Function that computes Brier scores for provided data points

    Parameters
    ----------
        y_true: array_like
            true labels (prior to one-hot encoding)

        pred_probs: array_like
            array of predicted probabilities

        n_classes: int
            number of classes in a problem


    Returns
    -------  

    """
    lab_enc = OneHotEncoder().fit(np.arange(n_classes).reshape(-1, 1))

    one_hot_true_labels = lab_enc.transform(y_true.reshape(-1, 1)).toarray()

    return np.sum((one_hot_true_labels - pred_probs)**2, axis=1) / 2


def top_label_brier_scores(y_true, pred_probs, n_classes=2, loss_type='squared'):
    """
    Function that computes top-label Brier scores for provided data points

    Parameters
    ----------
        y_true: array_like
            true labels (prior to one-hot encoding)

        pred_probs: array_like
            array of predicted probabilities

        n_classes: int
            number of classes in a problem


    Returns
    -------  

    """
    lab_enc = OneHotEncoder().fit(np.arange(n_classes).reshape(-1, 1))

    # make predictions
    pred_classes = pred_probs.argmax(axis=1)

    # encode as a binary mask
    one_hot_pred_classes = lab_enc.transform(
        pred_classes.reshape(-1, 1)).toarray()

    # sub-select parts corresponding to the top-predicted class
    top_pred_proba = pred_probs[one_hot_pred_classes.astype('bool')]
    cor_vs_inc_pred_class = (pred_classes == y_true).astype('int')

    if loss_type == 'squared':
        return (top_pred_proba-cor_vs_inc_pred_class)**2
    elif loss_type == 'absolute':
        return abs(top_pred_proba-cor_vs_inc_pred_class)


def true_class_brier_scores(y_true, pred_probs, n_classes=2, loss_type='squared'):
    """
    Function that computes true-class brier scores for provided data points

    Parameters
    ----------

    Returns
    ------- 
    """
    lab_enc = OneHotEncoder().fit(np.arange(n_classes).reshape(-1, 1))

    # encode as a binary mask
    one_hot_true_classes = lab_enc.transform(
        y_true.reshape(-1, 1)).toarray()

    # sub-select parts corresponding to the top-predicted class
    true_pred_proba = pred_probs[one_hot_true_classes.astype('bool')]

    if loss_type == 'squared':
        return (true_pred_proba-1)**2
    elif loss_type == 'absolute':
        return abs(true_pred_proba-1)


class Drop_tester(object):
    """
    Object that performs testing for accuracy drop
    """

    def __init__(self):

        # params for the testing procedure
        self.eps_tol = 0
        # set default boundedness params
        self.risk_lower_bound = 0
        self.risk_upper_bound = 1
        # absolute vs relative
        self.change_type = 'absolute'

        # params for source risk esimation
        self.source_delta = 0.025
        self.source_conc_type = 'betting'
        self.source_emp_risk = None
        self.source_risk_upper_bound = None
        self.source_num_of_samples_used = None
        self.source_rejection_threshold = None

        # params for target risk estimation
        self.target_delta = 0.025
        self.target_conc_type = 'betting'
        self.target_emp_risk = None
        self.target_risk_lower_bound = None
        self.target_num_of_samples_used = None

    def estimate_risk_source(self, ind_losses):
        """
        Estimate upper bound on the (source) risk given a labeled sample

        Parameters
        ----------

        Returns
        ------- 
        """

        self.source_num_of_samples_used = len(ind_losses)

        self.source_emp_risk = np.mean(ind_losses)

        # rescale if needed so that the losses lie in [0,1]
        scaled_losses = (ind_losses-self.risk_lower_bound) / \
            (self.risk_upper_bound-self.risk_lower_bound)

        if self.source_conc_type == 'hoeffding':
            self.source_risk_upper_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * min(hoeffding_ci_upper_limit(
                    scaled_losses, self.source_delta), 1)

        if self.source_conc_type == 'pm_bernstein':
            self.source_risk_upper_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * min(pm_bernstein_ci_upper_limit(
                    scaled_losses, self.source_delta), 1)

        if self.source_conc_type == 'betting':
            self.source_risk_upper_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * min(betting_ci_upper_limit(
                    scaled_losses, self.source_delta), 1)

        # define the threshold for the test
        if self.change_type == 'absolute':
            self.source_rejection_threshold = self.source_risk_upper_bound + self.eps_tol
        elif self.change_type == 'relative':
            self.source_rejection_threshold = (
                1 + self.eps_tol)*self.source_risk_upper_bound

    def estimate_risk_target(self, ind_losses):
        """
        Estimate lower bound on the (target) risk given a labeled sample

        Parameters
        ----------

        Returns
        ------- 
        """

        self.target_num_of_samples_used = len(ind_losses)

        self.target_emp_risk = np.mean(ind_losses)

        # rescale if needed so that the losses lie in [0,1]
        scaled_losses = (ind_losses-self.risk_lower_bound) / \
            (self.risk_upper_bound-self.risk_lower_bound)

        if self.target_conc_type == 'pm_hoeffding':
                # -1 to take the largest value
            self.target_risk_lower_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * max(pm_hoeffding_lower_limit(
                    scaled_losses, self.target_delta), 0)

        if self.target_conc_type == 'pm_bernstein':
            self.target_risk_lower_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * max(pm_bernstein_lower_limit(
                    scaled_losses, self.target_delta), 0)

        if self.target_conc_type == 'betting':
            self.target_risk_lower_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * max(betting_cs_lower_limit(
                    scaled_losses, self.target_delta), 0)

        if self.target_conc_type == 'conj-hoef':
            self.target_risk_lower_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * max(conjmix_hoeffding_cs(
                    scaled_losses, self.target_num_of_samples_used, alpha=2*self.target_delta)[0][-1], 0)

        if self.target_conc_type == 'conj-bern':
            self.target_risk_lower_bound = self.risk_lower_bound +\
                (self.risk_upper_bound-self.risk_lower_bound) * max(conjmix_empbern_cs(
                    scaled_losses, self.target_num_of_samples_used/4, alpha=2*self.target_delta)[0][-1], 0)

        # if self.risk_type == 'misclas':
        #     # if self.target_emp_risk is None:
        #     y_pred = self.model.predict(X_target)
        #     self.target_num_of_samples_used = len(y_target)
        #     errors_vec = y_pred != y_target
        #     self.target_emp_risk = np.mean(errors_vec)

        #     if self.target_conc_type == 'pm_hoeffding':
        #         # -1 to take the largest value
        #         self.target_risk_lower_bound = max(pm_hoeffding_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'pm_bernstein':
        #         self.target_risk_lower_bound = max(pm_bernstein_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'betting':
        #         self.target_risk_lower_bound = max(betting_cs_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'conj-hoef':
        #         self.target_risk_lower_bound = max(conjmix_hoeffding_cs(
        #             errors_vec, self.target_num_of_samples_used, alpha=2*self.target_delta)[0][-1], 0)

        #     if self.target_conc_type == 'conj-bern':
        #         self.target_risk_lower_bound = max(conjmix_empbern_cs(
        #             errors_vec, self.target_num_of_samples_used/4, alpha=2*self.target_delta)[0][-1], 0)

        # if self.risk_type == 'brier':
        #     # make predictions on a held-out set
        #     y_pred = self.model.predict_proba(X_target)
        #     self.target_num_of_samples_used = len(y_pred)
        #     errors_vec = brier_scores(y_target, y_pred)
        #     self.target_emp_risk = np.mean(errors_vec)

        #     if self.target_conc_type == 'pm_hoeffding':
        #         # -1 to take the largest value
        #         self.target_risk_lower_bound = max(pm_hoeffding_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'pm_bernstein':
        #         self.target_risk_lower_bound = max(pm_bernstein_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'betting':
        #         self.target_risk_lower_bound = max(betting_cs_lower_limit(
        #             errors_vec, self.target_delta), 0)

        #     if self.target_conc_type == 'conj-hoef':
        #         self.target_risk_lower_bound = max(conjmix_hoeffding_cs(
        #             errors_vec, self.target_num_of_samples_used, alpha=2*self.target_delta)[0][-1], 0)

        #     if self.target_conc_type == 'conj-bern':
        #         self.target_risk_lower_bound = max(conjmix_empbern_cs(
        #             errors_vec, self.target_num_of_samples_used/4, alpha=2*self.target_delta)[0][-1], 0)

    def test_for_drop(self):
        """
        Function to perform the test given estimates for the risk on the source and target

        Parameters
        ----------

        Returns
        ------- 
        """
        # check that both estimates have been computed
        if self.source_rejection_threshold is None:
            raise ValueError('Risk on the source has to be estimated')
        if self.target_risk_lower_bound is None:
            raise ValueError('Risk on the target has to be estimated')

        if self.target_risk_lower_bound > self.source_rejection_threshold:
            return True

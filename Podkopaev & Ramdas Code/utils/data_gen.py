import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


def compute_bayes_risk_binary(pi_1, mu_1, mu_0):
    """
    Compute risk of the Bayes predictor  
    """
    # reshape if needed
    local_mu_1 = mu_1.reshape(-1, 1)
    local_mu_0 = mu_0.reshape(-1, 1)
    # compute value at which to compute CDF
    thrshld = np.log(
        (1 - pi_1) / pi_1) + 1 / 2 * (np.linalg.norm(local_mu_1)**2 -
                                      np.linalg.norm(local_mu_0)**2)
    # compute params of Gaussians
    std_param = np.linalg.norm(local_mu_1 - local_mu_0)
    mean_param_0 = (local_mu_1 - local_mu_0).T.dot(local_mu_0)
    mean_param_1 = (local_mu_1 - local_mu_0).T.dot(local_mu_1)
    # compute risk
    misclas_prob_cond_0 = 1 - norm.cdf(thrshld, mean_param_0, std_param)
    misclas_prob_cond_1 = norm.cdf(thrshld, mean_param_1, std_param)

    bayes_risk = misclas_prob_cond_0 * (1 - pi_1) + misclas_prob_cond_1 * pi_1
    return bayes_risk


def compute_bayes_risk_binary_label_shift(pi_1_source, pi_1_target, mu_1, mu_0):
    """
    Compute risk of the Bayes predictor when label shift on the target is present 
    """
    # reshape if needed
    local_mu_1 = mu_1.reshape(-1, 1)
    local_mu_0 = mu_0.reshape(-1, 1)
    # compute value at which to compute CDF
    thrshld = np.log((1 - pi_1_source) / pi_1_source) + 1 / 2 * (
        np.linalg.norm(local_mu_1)**2 - np.linalg.norm(local_mu_0)**2)
    # compute params of Gaussians
    std_param = np.linalg.norm(local_mu_1 - local_mu_0)
    mean_param_0 = (local_mu_1 - local_mu_0).T.dot(local_mu_0)
    mean_param_1 = (local_mu_1 - local_mu_0).T.dot(local_mu_1)
    # compute risk
    misclas_prob_cond_0 = 1 - norm.cdf(thrshld, mean_param_0, std_param)
    misclas_prob_cond_1 = norm.cdf(thrshld, mean_param_1, std_param)

    bayes_risk = misclas_prob_cond_0 * (
        1 - pi_1_target) + misclas_prob_cond_1 * pi_1_target
    return bayes_risk


def generate_2d_example(pi_1, mu_0, mu_1, n_samples):
    """
    Function that simulates data for calibration example 

    Parameters
    ----------
        pi_1: float in (0,1)
            probabilities of class 1 labels

        mu_0: array_like
            cluster center for class 0

        mu_1: array_like
            cluster center for class 1

        n_samples: int
            size of the sample

    Returns
    -------
        features, labels: array_like
            sampled dataset
    """

    # generate labels
    labels = np.random.binomial(n=1, p=pi_1, size=n_samples)

    cov_mat = np.array([[1, 0], [0, 1]])

    # placeholders for features
    features = np.zeros([n_samples, 2])

    # generate features
    features[labels == 0] = np.random.multivariate_normal(
        mean=mu_0, cov=cov_mat, size=int(sum(labels == 0)))
    features[labels == 1] = np.random.multivariate_normal(
        mean=mu_1, cov=cov_mat, size=int(sum(labels == 1)))

    return features, labels


def generate_2d_4_class_example(probs_vec, mu_0, mu_1, mu_2, mu_3, n_samples):
    """
    Function that simulates data for calibration example 

    Parameters
    ----------
        pi_1: float in (0,1)
            probabilities of class 1 labels

        mu_i: array_like
            cluster center for class i, i=1,2,3,4

        n_samples: int
            size of the sample

    Returns
    -------
        features, labels: array_like
            sampled dataset
    """

    labels = np.random.multinomial(n=1, pvals=probs_vec, size=n_samples)

    labels_encoded = np.argmax(labels, axis=1)
    cov_mat = np.array([[1, 0], [0, 1]])

    # placeholders for features
    features = np.zeros([n_samples, 2])

    # generate features
    features[labels_encoded == 0] = np.random.multivariate_normal(
        mean=mu_0, cov=cov_mat, size=int(sum(labels_encoded == 0)))
    features[labels_encoded == 1] = np.random.multivariate_normal(
        mean=mu_1, cov=cov_mat, size=int(sum(labels_encoded == 1)))
    features[labels_encoded == 2] = np.random.multivariate_normal(
        mean=mu_2, cov=cov_mat, size=int(sum(labels_encoded == 2)))
    features[labels_encoded == 3] = np.random.multivariate_normal(
        mean=mu_3, cov=cov_mat, size=int(sum(labels_encoded == 3)))

    return features, labels_encoded


class LDA_predictor(object):
    def __init__(self):
        self.mean_class_0 = None
        self.mean_class_1 = None
        self.class_0_prior = None
        self.class_1_prior = None
        self.predict_both_classes = False

    def predict_proba(self, X):
        probs = X.dot(self.mean_class_0-self.mean_class_1) + 1/2 * \
            (np.linalg.norm(self.mean_class_1)**2 -
             np.linalg.norm(self.mean_class_0)**2)

        class_1_probs = 1/(self.class_0_prior /
                           self.class_1_prior * np.exp(probs)+1)

        if self.predict_both_classes:
            return np.vstack([1-class_1_probs, class_1_probs]).T

        return class_1_probs

    def predict(self, X):
        pred_probs = self.predict_proba(X)

        if self.predict_both_classes:
            return pred_probs[:, 1] >= 1/2

        return pred_probs >= 1/2

    def score(self, X, y):
        """evaluate accuracy"""
        y_pred = self.predict(X)

        return np.mean(y == y_pred)


class four_classes_LDA_predictor(object):
    """
    Object that represents Bayes-optimal decision rule for the Gaussian case
    """

    def __init__(self):
        mean_class_0 = None
        mean_class_1 = None
        mean_class_2 = None
        mean_class_3 = None
        cov_mat = None
        class_prop = None

    def predict_proba(self, X):
        """
        Make probabilistic prediction
        """
        prob_class_0 = multivariate_normal.pdf(
            X, self.mean_class_0, self.cov_mat)
        prob_class_1 = multivariate_normal.pdf(
            X, self.mean_class_1, self.cov_mat)
        prob_class_2 = multivariate_normal.pdf(
            X, self.mean_class_2, self.cov_mat)
        prob_class_3 = multivariate_normal.pdf(
            X, self.mean_class_3, self.cov_mat)

        bayes_rule_class_0 = self.class_prop[0]*prob_class_0/(
            self.class_prop[0]*prob_class_0 + self.class_prop[1]*prob_class_1 +
            self.class_prop[2]*prob_class_2 + self.class_prop[3]*prob_class_3)
        bayes_rule_class_1 = self.class_prop[1]*prob_class_1/(
            self.class_prop[0]*prob_class_0 + self.class_prop[1]*prob_class_1 +
            self.class_prop[2]*prob_class_2 + self.class_prop[3]*prob_class_3)
        bayes_rule_class_2 = self.class_prop[2]*prob_class_2/(
            self.class_prop[0]*prob_class_0 + self.class_prop[1]*prob_class_1 +
            self.class_prop[2]*prob_class_2 + self.class_prop[3]*prob_class_3)
        bayes_rule_class_3 = self.class_prop[3]*prob_class_3/(
            self.class_prop[0]*prob_class_0 + self.class_prop[1]*prob_class_1 +
            self.class_prop[2]*prob_class_2 + self.class_prop[3]*prob_class_3)

        probs = np.vstack(
            [bayes_rule_class_0, bayes_rule_class_1, bayes_rule_class_2, bayes_rule_class_3]).T

        return probs

    def predict(self, X):
        """
        Make prediction
        """
        probs = self.predict_proba(X)

        return probs.argmax(axis=1)

    def score(self, X, y):
        """
        Evaluate accuracy of the Bayes-optimal rule 
        """
        return np.mean(self.predict(X) == y)

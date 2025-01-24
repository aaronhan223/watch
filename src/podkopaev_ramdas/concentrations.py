import numpy as np
from numpy.core.numeric import indices

## Note from Drew: Commented out following import from confseq for now since not sure what library/package etc that is
#from confseq.boundaries import normal_mixture_bound, gamma_exponential_mixture_bound

# HOEFFDING'S CIs and CSs


def phi_h(lmbd): return lmbd**2 / 8


def hoeffding_ci_lower_limit(seq, delta):
    """
    Compute lower limit of the fixed-time Hoeffding's CI

    Parameters
    ----------

    Returns
    ------- 
    """
    n = len(seq)

    emp_mean = np.mean(seq)
    return emp_mean - np.sqrt(np.log(1/delta)/(2*n))


def hoeffding_ci_upper_limit(seq, delta):
    """
    Compute lower limit of the fixed-time Hoeffding's CI

    Parameters
    ----------

    Returns
    ------- 
    """
    n = len(seq)

    emp_mean = np.mean(seq)
    return emp_mean + np.sqrt(np.log(1/delta)/(2*n))


def pm_hoeffding_lower_limit(seq, delta, take_run_inter=True, last_step_only=True):
    """
    Compute lower limit of the fixed-time Hoeffding's CI

    Parameters
    ----------

    Returns
    ------- 
    """
    n = len(seq)

    obs_enum = np.arange(n) + 1

    pred_mixt = np.minimum(
        np.sqrt(8*np.log(1/delta)/(obs_enum*np.log(obs_enum+1))), 1)

    cumsum_pred_mixt = np.cumsum(pred_mixt)

    cumsum_cross_prod = np.cumsum(seq*pred_mixt)

    term_2 = np.log(1/delta) + np.cumsum(phi_h(pred_mixt))

    lower_lims = [(cumsum_cross_prod[i]-term_2[i]
                   )/cumsum_pred_mixt[i] for i in range(n)]

    if last_step_only:
        if take_run_inter:
            return max(lower_lims)

    # take running intersections
    if take_run_inter:
        # inters_lower_lims = [max(lower_lims[:i+1]) for i in range(n)]
        # return inters_lower_lims
        return np.maximum.accumulate(lower_lims)
    return lower_lims


def pm_hoeffding_upper_limit(seq, delta, take_run_inter=True, last_step_only=True):
    """
    Compute upper limit of the fixed-time Hoeffding's CI

    Parameters
    ----------

    Returns
    ------- 
    """
    n = len(seq)

    obs_enum = np.arange(n) + 1

    pred_mixt = np.minimum(
        np.sqrt(8*np.log(1/delta)/(obs_enum*np.log(obs_enum+1))), 1)

    cumsum_pred_mixt = np.cumsum(pred_mixt)

    cumsum_cross_prod = np.cumsum(seq*pred_mixt)

    term_2 = np.log(1/delta) + np.cumsum(phi_h(pred_mixt))

    upper_lims = [(cumsum_cross_prod[i]+term_2[i]
                   )/cumsum_pred_mixt[i] for i in range(n)]

    # take running intersections

    if last_step_only:
        if take_run_inter:
            return min(upper_lims)

    if take_run_inter:
        # inters_upper_lims = [min(upper_lims[:i+1]) for i in range(n)]
        return np.minimum.accumulate(upper_lims)

    return upper_lims

# EMPIRICAL BERNSTEIN'S CIs and CSs


def phi_e(lmbd): return (-np.log(1 - lmbd) - lmbd) / 4


def update_running_mean(x, num_of_obs, prev_mean_val=None):
    """
    Compute running mean of observations with an extra fake observation

    Parameters
    ----------

    Returns
    ------- 
    """
    # if the observation if the first one in a sequece
    if prev_mean_val == None:
        return (x+1/2)/2

    # fake observation is counted
    prev_sum = prev_mean_val * num_of_obs
    prev_sum += x
    return prev_sum/(num_of_obs+1)


def update_running_var(x, running_mean, num_of_obs, prev_var_val=None):
    """
    compute running variance

    Parameters
    ----------

    Returns
    ------- 
    """
    # if called for the first time
    if prev_var_val == None:
        return (1/4+(x-running_mean)**2)/2

    prev_sum = prev_var_val * num_of_obs
    prev_sum += (x-running_mean)**2
    return prev_sum/(num_of_obs+1)


def pm_bernstein_ci_lower_limit(seq, delta, c=1/2):
    """
    Compute lower limit of the predictably-mixed Hoeffding's CS

    Parameters
    ----------

    Returns
    ------- 
    """

    # compute the length of a sequence
    n = len(seq)

    # compute running means and variances

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    # compute values of pred mixture
    pred_mixtrure = np.array(
        [min(np.sqrt(2*np.log(1/delta)/(n*cur_var)), c) for cur_var in run_vars[:-1]])

    variances = np.array([4*(seq[i]-run_means[i])**2 for i in range(n)])

    term_2 = np.cumsum(variances*phi_e(pred_mixtrure)) + np.log(1/delta)

    cumsum_pred_mixture = np.cumsum(pred_mixtrure)

    cross_prod = seq*pred_mixtrure

    cumsum_cross_prod = np.cumsum(cross_prod)

    lower_lims = [(cumsum_cross_prod[i]-term_2[i]
                   )/cumsum_pred_mixture[i] for i in range(n)]

    return max(lower_lims)


def pm_bernstein_ci_upper_limit(seq, delta, c=1/2):
    """
    Compute upper limit of the predictably-mixed Hoeffding's CS

    Parameters
    ----------

    Returns
    ------- 
    """

    # compute the length of a sequence
    n = len(seq)

    # compute running means and variances

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    # compute values of pred mixture
    pred_mixtrure = np.array(
        [min(np.sqrt(2*np.log(1/delta)/(n*cur_var)), c) for cur_var in run_vars[:-1]])

    variances = np.array([4*(seq[i]-run_means[i])**2 for i in range(n)])

    term_2 = np.cumsum(variances*phi_e(pred_mixtrure)) + np.log(1/delta)

    cumsum_pred_mixture = np.cumsum(pred_mixtrure)

    cross_prod = seq*pred_mixtrure

    cumsum_cross_prod = np.cumsum(cross_prod)

    upper_lims = [(cumsum_cross_prod[i]+term_2[i]
                   ) / cumsum_pred_mixture[i] for i in range(n)]

    return min(upper_lims)


def pm_bernstein_lower_limit(seq, delta, c=1/2, take_run_inter=True, last_step_only=True):
    """
    Compute lower limit of the predictably-mixed Hoeffding's CS

    Parameters
    ----------

    Returns
    ------- 
    """

    # compute the length of a sequence
    n = len(seq)

    # compute running means and variances

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    # compute values of pred mixture

    pred_mixtrure = np.array(
        [min(np.sqrt(2*np.log(1/delta)/((i+1)*np.log(i+2)*run_vars[i])), c) for i in range(n)])

    variances = np.array([4*(seq[i]-run_means[i])**2 for i in range(n)])

    term_2 = np.cumsum(variances*phi_e(pred_mixtrure)) + np.log(1/delta)

    cumsum_pred_mixture = np.cumsum(pred_mixtrure)

    cross_prod = seq*pred_mixtrure

    cumsum_cross_prod = np.cumsum(cross_prod)

    lower_lims = [(cumsum_cross_prod[i]-term_2[i]
                   )/cumsum_pred_mixture[i] for i in range(n)]

    # take running intersections

    if last_step_only:
        if take_run_inter:
            return max(lower_lims)

    if take_run_inter:
        return np.maximum.accumulate(lower_lims)

    return lower_lims


def pm_bernstein_upper_limit(seq, delta, c=1/2, take_run_inter=True, last_step_only=True):
    """
    Compute upper limit of the predictably-mixed Hoeffding's CS

    Parameters
    ----------

    Returns
    ------- 
    """

    # compute the length of a sequence
    n = len(seq)

    # compute running means and variances

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    # compute values of pred mixture
    pred_mixtrure = np.array(
        [min(np.sqrt(2*np.log(1/delta)/((i+1)*np.log(i+2)*run_vars[i])), c) for i in range(n)])

    variances = np.array([4*(seq[i]-run_means[i])**2 for i in range(n)])

    term_2 = np.cumsum(variances*phi_e(pred_mixtrure)) + np.log(1/delta)

    cumsum_pred_mixture = np.cumsum(pred_mixtrure)

    cross_prod = seq*pred_mixtrure

    cumsum_cross_prod = np.cumsum(cross_prod)

    upper_lims = [(cumsum_cross_prod[i]+term_2[i]
                   )/cumsum_pred_mixture[i] for i in range(n)]

    # take running intersections

    if last_step_only:
        if take_run_inter:
            return min(upper_lims)

    if take_run_inter:
        return np.minimum.accumulate(upper_lims)

    return upper_lims


# Betting-based CI/CS

def betting_ci_lower_limit(seq, delta, c=1/2, resolution=200):
    """
    Perform bisection step in order to approximate L_bet

    Parameters
    ----------
        seq: array_like
            the original sequece of observations

        delta: float
            miscoverage probability

        resolution: int
            number of subintervals of [0,1] to create and use

    Returns
    -------  


    """

    # store cand lower bounds
    cand_lower_bounds = list()

    n = len(seq)

    mart_vals = np.zeros(shape=resolution)

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    cand_means = np.linspace(1e-5, 1-1e-5, num=resolution)

    for i in range(n):
        pred_mixtrure_plus = np.minimum(
            np.sqrt(2*np.log(1/delta)/(n*run_vars[i])), c/cand_means)
        mart_vals += np.log(1 + pred_mixtrure_plus *
                            (seq[i]-cand_means))
        cand_lower_bounds += [
            cand_means[np.argmax(mart_vals < np.log(1/delta))]]
    return max(cand_lower_bounds)


def betting_ci_upper_limit(seq, delta, c=1/2, resolution=200):
    """
    Perform bisection step in order to approximate U_bet

    Parameters
    ----------
        seq: array_like
            the original sequece of observations

        delta: float
            miscoverage probability

        resolution: int
            number of subintervals of [0,1] to create and use

    Returns
    -------  


    """

    # store cand lower bounds
    cand_upper_bounds = list()

    n = len(seq)

    mart_vals = np.zeros(shape=resolution)

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    # for bounded in [0,1]
    cand_means = np.linspace(1e-5, 1-1e-5, num=resolution)

    for i in range(n):

        pred_mixtrure_minus = np.minimum(
            np.sqrt(2*np.log(1/delta)/(n*run_vars[i])), c/(1-cand_means))
        mart_vals += np.log(1 - pred_mixtrure_minus *
                            (seq[i]-cand_means))
        # find the last occurence of when the condition is satisfied
        b = (mart_vals < np.log(1/delta))[::-1]
        i = resolution - np.argmax(b) - 1
        cand_upper_bounds += [
            cand_means[i]]

    return min(cand_upper_bounds)


def betting_cs_lower_limit(seq, delta, c=1/2, resolution=200, take_run_inter=True, last_step_only=True):
    """
    Perform bisection step in order to approximate L_bet

    Parameters
    ----------
        seq: array_like
            the original sequece of observations

        delta: float
            miscoverage probability

        resolution: int
            number of subintervals of [0,1] to create and use

    Returns
    -------  


    """

    # store cand lower bounds
    cand_lower_bounds = list()

    n = len(seq)

    mart_vals = np.zeros(shape=resolution)

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    cand_means = np.linspace(1e-5, 1-1e-5, num=resolution)

    for i in range(n):
        pred_mixtrure_plus = np.minimum(
            np.sqrt(2*np.log(1/delta)/((i+1)*np.log(i+2)*run_vars[i])), c/cand_means)
        mart_vals += np.log(1 + pred_mixtrure_plus *
                            (seq[i]-cand_means))
        cand_lower_bounds += [
            cand_means[np.argmax(mart_vals < np.log(1/delta))]]

    if last_step_only:
        if take_run_inter:
            return max(cand_lower_bounds)

    if take_run_inter:
        return np.maximum.accumulate(cand_lower_bounds)

    return cand_lower_bounds


def betting_cs_upper_limit(seq, delta, c=1/2, resolution=200, take_run_inter=True, last_step_only=True):
    """
    Perform bisection step in order to approximate U_bet

    Parameters
    ----------
        seq: array_like
            the original sequece of observations

        delta: float
            miscoverage probability

        resolution: int
            number of subintervals of [0,1] to create and use

    Returns
    -------  


    """

    # store cand lower bounds
    cand_upper_bounds = list()

    n = len(seq)

    mart_vals = np.zeros(shape=resolution)

    run_means = [1/2]
    run_vars = [1/4]

    obs_enum = np.arange(n) + 1

    for i in range(n):
        run_means += [update_running_mean(seq[i],
                                          obs_enum[i], prev_mean_val=run_means[-1])]

    for i in range(n):
        # account for the zero's entry in a list
        run_vars += [update_running_var(seq[i], run_means[i+1],
                                        obs_enum[i], prev_var_val=run_vars[-1])]

    cand_means = np.linspace(1e-5, 1-1e-5, num=resolution)

    for i in range(n):

        pred_mixtrure_minus = np.minimum(
            np.sqrt(2*np.log(1/delta)/((i+1)*np.log(i+2)*run_vars[i])), c/(1-cand_means))
        mart_vals += np.log(1 - pred_mixtrure_minus *
                            (seq[i]-cand_means))
        # print(mart_vals)
        # find the last occurence of the max value
        b = (mart_vals < np.log(1/delta))[::-1]
        i = len(b) - np.argmax(b) - 1
        cand_upper_bounds += [
            cand_means[i]]

    if last_step_only:
        if take_run_inter:
            return min(cand_upper_bounds)

    if take_run_inter:
        return np.minimum.accumulate(cand_upper_bounds)

    return cand_upper_bounds


# CONJUGATE MIXTURES (taken from the repo)

def conjmix_hoeffding_cs(x, t_opt, alpha=0.05, running_intersection=True):
    """
    Conjugate mixture Hoeffding confidence sequence
    Parameters
    ----------
    x, array-like of reals
        The observed data
    t_opt, positive real
        Time at which to optimize the confidence sequence
    alpha, (0, 1)-valued real
        Significance level
    running_intersection, boolean
        Should the running intersection be taken?
    Returns
    -------
    l, array-like of reals
        Lower confidence sequence
    u, array-like of reals
        Upper confidence sequence
    """
    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.cumsum(x) / t

    bdry = (normal_mixture_bound(t / 4,
                                 alpha=alpha,
                                 v_opt=t_opt / 4,
                                 alpha_opt=alpha,
                                 is_one_sided=False) / t)
    l, u = mu_hat_t - bdry, mu_hat_t + bdry

    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u


def conjmix_empbern_cs(x, v_opt, alpha=0.05, running_intersection=True):
    """
    Conjugate mixture empirical Bernstein confidence sequence
    Parameters
    ----------
    x, array-like of reals
        The observed data
    v_opt, positive real
        Intrinsic time at which to optimize the confidence sequence.
        For example, if the variance is given by sigma, and one
        wishes to optimize for time t, then v_opt = t*sigma^2.
    alpha, (0, 1)-valued real
        Significance level
    running_intersection, boolean
        Should the running intersection be taken?
    Returns
    -------
    l, array-like of reals
        Lower confidence sequence
    u, array-like of reals
        Upper confidence sequence
    """
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    mu_hat_t = S_t / t
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0:(len(mu_hat_t) - 1)])
    V_t = np.cumsum(np.power(x - mu_hat_tminus1, 2))
    bdry = (gamma_exponential_mixture_bound(
        V_t, alpha=alpha / 2, v_opt=v_opt, c=1, alpha_opt=alpha / 2) / t)
    l, u = mu_hat_t - bdry, mu_hat_t + bdry
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)
    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u

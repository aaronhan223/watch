import numpy as np


def ville_procedure(p_values, threshold=100, verbose=False):
    """
    Implements the Ville procedure. Raises an alarm when the martingale exceeds the threshold.
    """
    martingale = 1.0  # Start with initial capital of 1
    for i, p in enumerate(p_values):
        # This implies that the martingale grows if the p-value is small (indicating that the observation is unlikely under the null hypothesis)
        # and shrinks if the p-value is large.
        martingale *= (1 / p)
        if martingale >= threshold and verbose:
            print(f"Alarm raised at observation {i + 1} with martingale value = {martingale}")
            # break
    return martingale

def cusum_procedure(S, alpha, verbose=False):
    """
    Implements the CUSUM statistic.
    """
    gamma = np.zeros(len(S))
    threshold = np.percentile(S, 100 * alpha)
    for n in range(1, len(S)):
        gamma[n] = max(S[n] / S[i] for i in range(n))
        if gamma[n] >= threshold and verbose:
            print(f"Alarm raised at observation {n} with gamma={gamma[n]}")
            # return True, gamma
    return False, gamma

def shiryaev_roberts_procedure(S, c, verbose=False):
    """
    Implements the Shiryaev-Roberts statistic.
    """
    sigma = np.zeros(len(S))
    for n in range(1, len(S)):
        sigma[n] = sum(S[n] / S[i] for i in range(n))
        if sigma[n] >= c and verbose:
            print(f"Alarm raised at observation {n} with sigma={sigma[n]}")
            # return True, sigma
    return False, sigma

## 20241203: Changed J from 0.01 to 0.05
def simple_jumper_martingale(p_values, J=0.01, threshold=100, verbose=False):
    """
    Implements the Simple Jumper martingale betting strategy.
    """
    C_minus1, C_0, C_1 = 1/3, 1/3, 1/3
    C = 1
    martingale_values = []

    for i, p in enumerate(p_values):
        C_minus1 = (1 - J) * C_minus1 + (J / 3) * C
        C_0 = (1 - J) * C_0 + (J / 3) * C
        C_1 = (1 - J) * C_1 + (J / 3) * C

        C_minus1 *= (1 + (p - 0.5) * -1)
        C_0 *= (1 + (p - 0.5) * 0)
        C_1 *= (1 + (p - 0.5) * 1)

        C = C_minus1 + C_0 + C_1
        martingale_values.append(C)

        if C >= threshold and verbose:
            print(f"Alarm raised at observation {i} with martingale value={C}")
            # return True, np.array(martingale_values)
    
    return False, np.array(martingale_values)


def composite_jumper_martingale(p_values, threshold=100, verbose=False):
    """
    Implements the Simple Jumper martingale betting strategy.
    """
    J_list = [0.0001, 0.001, 0.01, 0.1, 1]
    num_J = 5
    C_minus1, C_0, C_1 = np.repeat(1/3,num_J), np.repeat(1/3,num_J), np.repeat(1/3,num_J)
    C = 1
    martingale_values = []
    
    for i, p in enumerate(p_values):
        for J_i, J in enumerate(J_list):
            C_minus1[J_i] = (1 - J) * C_minus1[J_i] + (J / 3) * C
            C_0[J_i] = (1 - J) * C_0[J_i] + (J / 3) * C
            C_1[J_i] = (1 - J) * C_1[J_i] + (J / 3) * C

            C_minus1[J_i] *= (1 + (p - 0.5) * -1)
            C_0[J_i] *= (1 + (p - 0.5) * 0)
            C_1[J_i] *= (1 + (p - 0.5) * 1)

        C = np.mean([C_minus1[J_i] + C_0[J_i] + C_1[J_i] for J_i in range(num_J)])
        martingale_values.append(C)

        if C >= threshold and verbose:
            print(f"Alarm raised at observation {i} with martingale value={C}")
            # return True, np.array(martingale_values)
    
    return False, np.array(martingale_values)
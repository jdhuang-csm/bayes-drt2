import cvxopt
import numpy as np

from bayes_drt2.matrices import construct_L


def solve_s(m, x, sv, alpha, beta):
    """
    Determine optimal values of local penalty scale parameters, s_dm
    :param ndarray m: integrated penalty matrix
    :param ndarray x: coefficient vector
    :param ndarray sv: previous vector of local penalty scale parameters
    :param float alpha: effective alpha hyperparameter of gamma prior on s_dm. Must be > 1
    :param float beta: effective beta hyperparameter of gamma prior on s_dm. Must be > 0
    :return:
    """
    xm = np.diag(x)
    sm = np.diag(sv ** 0.5)
    xsm = xm @ sm @ m @ xm
    xsm = xsm - np.diag(np.diagonal(xsm))
    b1 = np.sum(xsm, axis=0)
    b2 = np.sum(xsm, axis=1)
    b = (b1 + b2) / 2

    # Exponential prior
    d = x ** 2 * np.diagonal(m) + beta
    sv = (2 * b ** 2 - 2 * abs(b) * np.sqrt(4 * d * (alpha - 1) + b ** 2) + 4 * d * (alpha - 1)) / (
            4 * d ** 2)
    return sv


def solve_p(m, x, sv, alpha, beta, xmx_norm):
    """
    Determine optimal value of p_d (weight of derivative of order d)
    :param ndarray m: integrated penalty matrix
    :param ndarray x: coefficient vector
    :param ndarray sv: vector of local penalty scale parameters
    :param float alpha: effective alpha hyperparameter of gamma prior on p_d. Must be > 1
    :param float beta: effective beta hyperparameter of gamma prior on p_d. Must be > 0
    :param xmx_norm: normalizing value of scalar x.T @ S @ M @ S @ x. Determined from ordinary ridge coefficients
    :return: float
    """
    sm = np.diag(sv ** 0.5)
    xsmsx = x.T @ sm @ m @ sm @ x
    # print('xsmsx:', xsmsx)
    return (alpha - 1) / (xsmsx / xmx_norm + beta)


def solve_init_weight_scale(w_scale_est, alpha, beta):
    b = (1/2 - alpha + 1)
    s_hat = (-b + np.sqrt(b ** 2 + 2 * beta * (w_scale_est ** -2))) / (2 * beta)
    w_hat = s_hat ** -0.5
    return w_hat


def construct_var_matrix(frequencies, vmm_epsilon, reim_cor, error_structure):
    n = len(frequencies)
    vmm = np.zeros((2 * n, 2 * n))
    if error_structure is None:
        vmm_main = construct_L(frequencies, epsilon=vmm_epsilon, order=0)
    elif error_structure == 'uniform':
        vmm_main = np.ones((n, n))
    # Diagonals: re-re and im-im averaging
    vmm[:n, :n] = vmm_main * (1 - reim_cor)
    vmm[n:, n:] = vmm_main * (1 - reim_cor)
    # Off-diagonals: re-im and im-re averaging
    vmm[n:, :n] = vmm_main * reim_cor
    vmm[:n, n:] = vmm_main * reim_cor
    vm_rowsum = np.sum(vmm, axis=1)
    vmm /= vm_rowsum[:, None]

    return vmm


def estimate_weights(x, y, vmm, a_re, a_im, est_weights=None, error_structure=None, w_alpha=None, w_beta=None):
    resid_re = a_re @ x - y.real
    resid_im = a_im @ x - y.imag

    n = len(y)
    # Concatenate residuals
    resid = np.concatenate((resid_re, resid_im))
    # Estimate error variance vectors from weighted mean of residuals
    s_hat = vmm @ resid ** 2  # concatenated variance vector
    s_hat_re = s_hat[:n]
    s_hat_im = s_hat[n:]

    # Convert variance to weights
    w_hat_re = s_hat_re ** -0.5
    w_hat_im = s_hat_im ** -0.5

    if est_weights is not None:
        # To ensure convergence (avoid poor initial fit resulting in weights going to zero), average current weights
        # with initial estimate from overfitted ridge

        # As current weights approach initial estimate (indicating better fit), give them more weight
        scale_current = np.mean(s_hat_re) ** -0.5 + np.mean(s_hat_im) ** -0.5
        scale_est = np.mean(est_weights.real ** -2) ** -0.5 + np.mean(est_weights.imag ** -2) ** -0.5
        # frac_current = 0.5
        frac_current = scale_current / (scale_current + scale_est)
        print('frac_current', frac_current)
        frac_est = 1 - frac_current
        # Take mean of current and initial weight estimates
        w_hat_re = (frac_current * w_hat_re + frac_est * est_weights.real)
        w_hat_im = (frac_current * w_hat_im + frac_est * est_weights.imag)
        # w_hat_im = w_hat_im * (w_im_scale + est_weight_scale.imag) / (2 * w_im_scale)

    if w_alpha is not None and w_beta is not None:
        # Apply prior
        w_scale_re = np.mean(w_hat_re)
        w_scale_im = np.mean(w_hat_im)
        w_hat_re = w_hat_re * solve_init_weight_scale(w_scale_re, w_alpha, w_beta) / w_scale_re
        w_hat_im = w_hat_im * solve_init_weight_scale(w_scale_im, w_alpha, w_beta) / w_scale_im

    return w_hat_re + 1j * w_hat_im


def initialize_weights(part, penalty_matrices, derivative_weights, p_vector, s_vectors, target_scaled,
                            a_re, a_im, vmm, l1_lambda_vector, nonneg, error_structure,
                            iw_alpha, iw_beta):
    # Calculate L2 penalty matrix (SMS)
    l2_matrices = [penalty_matrices[f'M{n}'] for n in range(0, 3)]
    sms = calculate_sms(np.array(derivative_weights) * p_vector, l2_matrices, s_vectors)
    sms *= 1e-6  # Apply very small penalty strength for overfit

    # Solve the ridge problem with QP: optimize x
    # Multiply sms by 2 due to exponential prior
    cvx_result = solve_convex_opt(part, target_scaled.real, target_scaled.imag, a_re, a_im, 2 * sms, l1_lambda_vector,
                                  nonneg)
    x_overfit = np.array(list(cvx_result['x']))

    # Get weight structure
    est_weights = estimate_weights(x_overfit, target_scaled, vmm, a_re, a_im, est_weights=None,
                                   error_structure=error_structure)

    # Get global weight scale
    # Need to average variance, not weights
    s_scale_re = np.mean(est_weights.real ** -2)
    s_scale_im = np.mean(est_weights.imag ** -2)
    w_scale_re = s_scale_re ** -0.5
    w_scale_im = s_scale_im ** -0.5

    est_weight_scale = w_scale_re + 1j * w_scale_im

    # Solve for initial weight scale
    # iw_beta = iw_0 ** 2 * (iw_alpha - 1.5)
    init_weight_scale_re = solve_init_weight_scale(w_scale_re, iw_alpha, iw_beta)
    init_weight_scale_im = solve_init_weight_scale(w_scale_im, iw_alpha, iw_beta)

    # Get initial weight vector (or scalar)
    init_weights = est_weights.real * init_weight_scale_re / w_scale_re \
                   + 1j * est_weights.imag * init_weight_scale_im / w_scale_im

    # init_weights = solve_init_weight_scale(est_weights.real, iw_alpha, iw_beta) \
    # + 1j * solve_init_weight_scale(est_weights.imag, iw_alpha, iw_beta)

    return est_weights, init_weights


def calculate_sms(derivative_weights, l2_matrices, s_vectors):
    sms = np.zeros_like(l2_matrices[0])
    for n, d_weight in enumerate(derivative_weights):
        if d_weight > 0:
            sv = s_vectors[n]
            sm = np.diag(sv ** 0.5)
            m = l2_matrices[n]
            sms += d_weight * sm @ m @ sm

    return sms


def solve_convex_opt(part, WZ_re, WZ_im, WA_re, WA_im, sms, l1_lambda_vector, nonneg):
    if part == 'both':
        P = cvxopt.matrix((WA_re.T @ WA_re + WA_im.T @ WA_im + sms).T)
        q = cvxopt.matrix((-WA_re.T @ WZ_re - WA_im.T @ WZ_im + l1_lambda_vector).T)
    elif part == 'real':
        P = cvxopt.matrix((WA_re.T @ WA_re + sms).T)
        q = cvxopt.matrix((-WA_re.T @ WZ_re + l1_lambda_vector).T)
    else:
        P = cvxopt.matrix((WA_im.T @ WA_im + sms).T)
        q = cvxopt.matrix((-WA_im.T @ WZ_im + l1_lambda_vector).T)

    G = cvxopt.matrix(-np.eye(WA_re.shape[1]))
    if nonneg:
        # coefficients must be >= 0
        h = cvxopt.matrix(np.zeros(WA_re.shape[1]))
    else:
        # coefficients can be positive or negative
        h = 10 * np.ones(WA_re.shape[1])
        # HFR and inductance must still be nonnegative
        h[0:2] = 0
        # print(h)
        h = cvxopt.matrix(h)
    # print('neg')

    return cvxopt.solvers.qp(P, q, G, h)
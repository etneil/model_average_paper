import numpy as np
import gvar as gv
import lsqfit

# Info criteria
def naive_IC_from_fit(fr):
    return fr.chi2


def BIC_from_fit(fr, ND):
    if ND is None:
        raise ValueError("Must specify ND to use Bayes IC!")
    return fr.chi2 + len(fr.p.keys()) * np.log(ND)


def AIC_from_fit(fr):
    return fr.chi2 + 2 * len(fr.p.keys())


def GAP_from_fit(fr):
    par_list = fr.p.keys()
    par_values = [fr.p[par] for par in par_list]
    Sigma_star = gv.evalcov(par_values)

    prior_values = [fr.prior[par] for par in par_list]
    Sigma_tilde = gv.evalcov(prior_values)

    return (
        fr.chi2
        - np.log(np.linalg.det(Sigma_star))
        + np.log(np.linalg.det(Sigma_tilde))
        + 2 * len(fr.p.keys())
    )


def get_raw_model_prob(fr, IC="AIC", N_cut=0, return_IC=False, ND=None, yraw=None):
    """
    Compute probability from log likelihood (LL) for a given fit.
    This is "raw" in the sense that it's not normalized over the
    space of all models, which should be done separately.

    Relation to info criteria:
        LL = -1/2 * IC

    Args:
      fr: Fit result object, from lsqfit module.
      IC: Which info criterion to use.  Options: AIC (default), BIC, GAP, BPIC, naive.
      N_cut: Correction for data subset selection.
      gamma_prior: Function with calculates the contribution to LL
                   from the presence of an indicator variable 'gamma'.
                   If 'None' (default), no gamma prior is included.
    """

    if IC == "BIC":
        LL = -0.5 * BIC_from_fit(fr, ND)
    elif IC == "AIC":
        LL = -0.5 * AIC_from_fit(fr)
    elif IC == "GAP":
        LL = -0.5 * GAP_from_fit(fr)
    elif IC == "naive":
        LL = -0.5 * naive_IC_from_fit(fr)
    else:
        raise ValueError(f"Unrecognized choice of info criterion: {IC}")

    # Correction to IC is +2 * N_cut - except for naive IC, which ignores this
    if IC != "naive":
        LL -= N_cut

    if return_IC:
        return LL, np.exp(LL)
    else:
        return np.exp(LL)


def model_avg(gv_list, pr_list):
    """
    Given a list of single-model expectation values {<f(a)>_M} as gvars,
    and a list of raw model probabilities, return the model-averaged estimate
    for <f(a)> as a gvar.
    """

    # Ensure model probabilities are normalized to 1
    pr_list /= np.sum(pr_list)

    mean_avg = np.sum(gv.mean(gv_list) * pr_list)
    var_avg = np.sum(gv.var(gv_list) * pr_list)
    var_avg += np.sum(gv.mean(gv_list) ** 2 * pr_list)
    var_avg -= (np.sum(gv.mean(gv_list) * pr_list)) ** 2

    return gv.gvar(mean_avg, np.sqrt(var_avg))


def obs_avg_full_width(obs_array, Q_array, fit_array, bf_i=None, p_min=0.1):
    """
    For comparison to model averaging, take the full width over all model results
    as an estimate of systematic error.

    Only model results with fit p-value greater than p_min are included.
    """

    Q_array = np.asarray(Q_array)
    obs_array = np.asarray(obs_array)

    milc_C_array = []
    for fr in fit_array:
        milc_C = fr.Q * fr.dof / np.sum([g.var for g in fr.p.values()])
        milc_C_array.append(milc_C)

    if bf_i is None:
        bf_i = np.argmax(milc_C_array)

    best_obs = obs_array[bf_i]

    good_obs = gv.mean(obs_array[Q_array > 0.1])

    try:
        syst_obs = gv.gvar(
            best_obs.mean, best_obs.sdev + (np.max(good_obs) - np.min(good_obs)) / 2
        )
    except ValueError:
        # If good_obs is empty, will throw ValueError, in which case
        # just return best_obs unmodified
        syst_obs = best_obs

    return syst_obs

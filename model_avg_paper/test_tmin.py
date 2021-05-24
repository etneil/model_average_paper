#!/usr/bin/env python3

from .stats import *
from .synth_data import *
from .fitting import *

def cut_data_tmin(data, tmin, tmax=None):
    """
    Cuts data to only include data above tmin
    (and below tmax, if provided.)

    Args:
      tmin: Minimum t-value to cut at (not inclusive, i.e.
            t > tmin is kept.)
      tmax: (optional) Maximum t-value to cut at (also not
            inclusive.)

    Returns:
      An abbreviated synth_data dictionary with the cuts applied.
    """
    T_cut = data['t'] > tmin
    if tmax:
        T_cut = np.logical_and(T_cut, data['t'] < tmax)

    return {
        't': data['t'][T_cut],
        'y': data['y'][T_cut],
        'yexact': data['yexact'][T_cut],
        'yraw': data['yraw'][:,T_cut],
        'ND': data['ND'],
    }


def test_vary_tmin_SE(
    data,
    Nt=32,
    max_tmin=28,
    obs_name="E0",
    IC="AIC",
    cross_val=False,
):
    """
    Test a fixed single-exponential model against the given data with a sliding t_min cut, extracting the given
    common fit parameter using model averaging.

    Args:
      data: Synthetic data set to run fits against.
      Nt: Parameter for exponential model/data

      max_tmin: Maximum value of tmin to use (minimum is 0.)
      obs_name: Common fit parameter to study with model averaging (default: 'a0', the y-intercept.)
      IC: Info criterion to use for model probability and averaging (default: AIC)

      cross_val: Use cross validation with a fraction of the data to set priors.  Assumes data is a list of
                two samples, with the "training" sample first.

    """

    # Need at least 1 dof to fit with
    assert max_tmin < Nt - 2

    if cross_val:
        train_data = data[0]
        test_data = data[1]
    else:
        test_data = data

    T_test = np.arange(0, Nt)

    # Run fits of synthetic data vs. tmin
    obs_vs_tmin = []
    prob_vs_tmin = []
    fit_vs_tmin = []
    fitIC_vs_tmin = []
    Q_vs_tmin = []

    for tmin in T_test[:max_tmin]:
        cut_test_data = cut_data_tmin(test_data, tmin)
        if cross_val:
            cut_train_data = cut_data_tmin(train_data, tmin)
            train_fit = run_fit_single_exp(cut_train_data, Nt=Nt)  
            this_fit = run_fit_single_exp(cut_test_data, Nt=Nt, priors_SE=train_fit.p)
        else:
            this_fit = run_fit_single_exp(cut_test_data, Nt=Nt)

        fit_vs_tmin.append(this_fit)
        obs_vs_tmin.append(this_fit.p[obs_name])

        fitIC, prob = get_raw_model_prob(
            this_fit, return_IC=True, IC=IC, ND=cut_test_data["ND"], N_cut=tmin, yraw=cut_test_data["yraw"]
        )

        prob_vs_tmin.append(prob)
        fitIC_vs_tmin.append(fitIC)
        Q_vs_tmin.append(this_fit.Q)

    # Compute model-averaged result
    obs_avg = model_avg(obs_vs_tmin, prob_vs_tmin)

    # Return results as a dictionary
    return {
        "tmin": T_test[:max_tmin],
        "data": data,
        "fits": fit_vs_tmin,
        "obs": obs_vs_tmin,
        "probs": prob_vs_tmin,
        "ICs": fitIC_vs_tmin,
        "obs_avg": obs_avg,
        "Qs": Q_vs_tmin,
    }

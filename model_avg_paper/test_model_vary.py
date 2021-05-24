#!/usr/bin/env python3

from .stats import *
from .synth_data import *
from .fitting import *


def test_vary_poly(
    test_data, Nt, m_max=6, m_min=1, prior_width=10, obs_name="a0", IC="AIC"
):
    """
    Test a varying set of polynomial models against the given data, extracting the given
    common fit parameter using model averaging.

    Args:
      test_data: Synthetic data set to run fits against.
      Nt: Normalizing constant for the polynomial model (typically Nt=maximum value of the 't' coordinate.)

      m_max: Max polynomial order to include (default: 6.)
      m_min: Min polynomial order to include (default: 1, i.e. linear.)
      prior_width: Width of priors for model fit parameters (default: 10.)
      obs_name: Common fit parameter to study with model averaging (default: 'a0', the y-intercept.)
      IC: Info criterion to use for model probability and averaging (default: AIC)
    """

    # Run fits
    obs_vs_Np = []
    prob_vs_Np = []
    fitIC_vs_Np = []
    fit_vs_Np = []
    Q_vs_Np = []

    for Np in range(m_min, m_max + 1):
        this_fit = run_fit_poly(test_data, Nt, Np, prior_width=prior_width)
        fit_vs_Np.append(this_fit)
        obs_vs_Np.append(this_fit.p[obs_name])
        fitIC, prob = get_raw_model_prob(
            this_fit, IC=IC, return_IC=True, ND=test_data["ND"], yraw=test_data["yraw"]
        )
        prob_vs_Np.append(prob)
        fitIC_vs_Np.append(fitIC)
        Q_vs_Np.append(this_fit.Q)

    obs_avg = model_avg(obs_vs_Np, prob_vs_Np)

    return {
        "m": np.array(range(m_min, m_max + 1)),
        "data": test_data,
        "fits": fit_vs_Np,
        "obs": obs_vs_Np,
        "probs": prob_vs_Np,
        "obs_avg": obs_avg,
        "Qs": Q_vs_Np,
        "ICs": fitIC_vs_Np,
    }

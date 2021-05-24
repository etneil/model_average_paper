#!/usr/bin/env python3

from .synth_data import *
import lsqfit

def run_fit_poly(
    data, Nt, m, prior_width=10, **kwargs,
):
    """
    Setup and run a fit against a polynomial model of order m.

    Args:
      data: Synthetic data set to run fits against.
      Nt: Normalizing constant for the polynomial model 
          (typically Nt=maximum value of the independent coordinate.)
      m: Order of the polynomial model to fit.
      prior_width: Width of priors for model fit parameters (default: 10).

    Returns:
      fr: an lsqfit FitResults object.

    kwargs are passed to the lsqfit.nonlinear_fit function.
    """

    priors_poly = {}
    for i in range(0, m + 1):
        priors_poly[f"a{i}"] = gv.gvar(0.0, prior_width)

    def fit_model(x, p):
        return poly_model(x, p, Nt=Nt, m=m)

    fr = lsqfit.nonlinear_fit(
        data=(data["t"], data["y"]),
        fcn=fit_model,
        prior=priors_poly,
        **kwargs
    )

    return fr

def run_fit_single_exp(
    data,
    Nt=32,
    priors_SE=None,
    **kwargs,
):
    """
    Setup and run a fit against the single-exponential model.

    Args:
      data: Synthetic data set to run fits against.
      Nt: "Finite volume" parameter for the exponential model
      priors_SE: dictionary of priors.  Overrides the default priors
                 used if equal to None.

    Returns:
      fr: an lsqfit FitResults object.

    kwargs are passed to the lsqfit.nonlinear_fit function.
    """

    if priors_SE is None:
        priors_SE = {
            "A0": gv.gvar("0(10)"),
            "E0": gv.gvar("1(1)"),
        }

    fr = lsqfit.nonlinear_fit(
        data=(data["t"], data["y"]),
        fcn=single_exp_model,
        prior=priors_SE,
        **kwargs
    )

    return fr





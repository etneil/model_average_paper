#!/usr/bin/env python3

import numpy as np
import gvar as gv


# Models, in the style of the 'lsqfit' Python module
def poly_model(t, p, Nt, m=2):
    ans = 0.0
    for i in range(0, m + 1):
        ans += p[f"a{i}"] * (t / Nt) ** i
    return ans

def single_exp_model(t, p):
    return p["A0"] * np.exp(-p["E0"] * t)


def multi_exp_model(t, p, Nexc=2):
    ans = 0.0
    for i in range(0, Nexc):
        ans += p["A{}".format(i)] * np.exp(-p["E{}".format(i)] * t)
    return ans


def gen_synth_data(
    t, p0, model, noise_amp=1.0, noise_samples=200, frac_noise=False
):
    """
    Given a model and some numeric parameters (defining the range of independent
    variables x and the 'model truth' p0), generates synthetic data equal to the model
    truth plus some noise.

    Args:
      t: NumPy array specifying the values of independent
         coordinates to sample the model at.
      p0: Dict of the form { par_name: <gv.gvar> } defining the "model truth" values of
          the model parameters.
      model: Model function that takes (x,p0) as inputs - see examples defined above.

      noise_amp: Optional float, determines the magnitude of the noise to be added.
      noise_samples: Optional int, determines the number of random noise samples
      frac_noise: Boolean flag, default False.  If True, the synthetic data takes
                  the form y_exact * (1 + noise), instead of y_exact + noise.

    """

    y_exact = model(t, p0)

    if frac_noise:
        noise = np.random.normal(1.0, noise_amp, (noise_samples, len(y_exact)))
        y_noisy = noise * y_exact
    else:
        y_noisy = np.random.normal(
            y_exact, noise_amp * np.ones_like(y_exact), (noise_samples, len(y_exact))
        )

    y = gv.dataset.avg_data(y_noisy)

    return {
        "t": t,
        "yexact": y_exact,
        "y": y,
        "yraw": y_noisy,
        "ND": noise_samples,  # Store sample size in case needed later, e.g. for BIC
    }


def gen_synth_data_corr(
    t,
    p0,
    model,
    rho,
    noise_amp=1.0,
    noise_samples=200,
    frac_noise=False,
    cross_val=False,
    cv_frac=0.1,
):
    """
    Given a model and some numeric parameters (defining the range of independent
    variables x and the 'model truth' p0), generates synthetic data equal to the
    model truth plus correlated noise.

    Args:
      t: NumPy array specifying the values of
         independent coordinates to sample the model at.
      p0: Dict of the form { par_name: <gv.gvar> } defining the "model truth" values
          of the model parameters.
      model: Model function that takes (x,p0) as inputs - see examples defined above.

      noise_amp: Optional float, determines the magnitude of the noise to be added.
      noise_samples: Optional int, determines the number of random noise samples to take
      frac_noise: Boolean flag, default False.  If True, the synthetic data takes the
                  form y_exact * (1 + noise), instead of y_exact + noise.
      cross_val: Boolean flag, default False.  If True, returns a list of two data
                 samples, with the first representing a "training" sample as a fraction
                 of the total number of samples requested.
      cv_frac: Optional float, between 0 and 1, default 0.1.  Determines the fraction
               of data to use in the training sample when doing cross-validation.
    """
    assert 0 < rho < 1
    assert 0 < cv_frac < 1

    y_exact = model(t, p0)

    # Construct noise array
    Ny = len(y_exact)
    noise_src = gv.gvar([(0, noise_amp)] * Ny)
    noise_corr = np.fromfunction(
        lambda i, j: rho ** (np.abs(i - j)), (Ny, Ny), dtype=np.float64
    )
    noise_src = gv.correlate(noise_src, noise_corr)

    noise_gen = gv.raniter(noise_src)
    noise_array = np.asarray([next(noise_gen) for i in range(noise_samples)])

    if frac_noise:
        y_noisy = y_exact * (1.0 + noise_array)
    else:
        y_noisy = y_exact + noise_array

    if cross_val:
        Ntrain = int(cv_frac * noise_samples)
        y1 = gv.dataset.avg_data(y_noisy[:Ntrain, :])
        y2 = gv.dataset.avg_data(y_noisy[Ntrain:, :])

        return [
            {
                "t": t,
                "yexact": y_exact,
                "yraw": y_noisy[:Ntrain, :],
                "ND": Ntrain,
                "y": y1,
            },
            {
                "t": t,
                "yexact": y_exact,
                "yraw": y_noisy[Ntrain:, :],
                "ND": noise_samples - Ntrain,
                "y": y2,
            },
        ]

    y = gv.dataset.avg_data(y_noisy)

    return {
        "t": t,
        "yexact": y_exact,
        "y": y,
        "yraw": y_noisy,
        "ND": noise_samples,  # Store sample size in case needed later, e.g. for BIC
    }


def cut_synth_data_Nsamp(synth_data, Ns_cut):
    """
    Given a synthetic data set, places a cut in the space of random samples, returning
    a reduced data set from the original.  For a cross-validation set, only cuts
    the testing data set, not the training data.

    Args:
      synth_data: A synthetic data set, produced by one of the gen_synth_data functions
        implemented above.
      Ns_cut: How many samples to keep in the cut data set.  Must be less than or equal
        to the number of samples ND in the original synthetic data.

    Returns:
      cut_data: A new synthetic data set formed from the first Ns_cut samples.
    """


    # Check if the synth_data is a list of 2 or just one
    if type(synth_data) is list:
        assert Ns_cut <= synth_data[1]["ND"]

        cut_data = []
        cut_data.append(synth_data[0])  # No cut on training data

        cut_data_raw = synth_data[1]["yraw"][:Ns_cut, :]
        cut_data_avg = gv.dataset.avg_data(cut_data_raw)

        cut_data.append({
            "t": synth_data[1]["t"],
            "yexact": synth_data[1]["yexact"],
            "ND": Ns_cut,
            "yraw": cut_data_raw,
            "y": cut_data_avg,
        })
    else:
        assert Ns_cut <= synth_data["ND"]

        cut_data_raw = synth_data["yraw"][:Ns_cut, :]
        cut_data_avg = gv.dataset.avg_data(cut_data_raw)

        cut_data = {
            "t": synth_data["t"],
            "yexact": synth_data["yexact"],
            "ND": Ns_cut,
            "yraw": cut_data_raw,
            "y": cut_data_avg,
        }

    return cut_data
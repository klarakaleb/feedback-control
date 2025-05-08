#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Setup up parameters to run network model.
"""

import numpy as np
import os, shutil


def main(
    savname,
    dataset_name="Reaching",
    dataset_name_AD=None,
    fb_density=1,
    fb_delay=0,
    rand_seed=0,
    protocol=None,
    rot_phi=30,
    freqshift=1,
    learning_rate=1e-3,
    record_gradients=False,
    record_jacobians=False,
    custom_delay=None,
    control_dim=1,
    n=None,
    record_weights=False,
    get_grads_per_example=False,
    record_lca=False,
    record_weights_norm=False,
    record_r_inputs=False,
    recurrent=True,
    error_type="error",
    error_detach=False,
    nlayers=1,
    batch_size=20,
    vel=10,
    go_to_peak=50,
    fb_freeze=False,
    wfb_frozen_phase=0,
    pratio = 0.25,
    freq = None,
):

    if dataset_name_AD is None:
        dataset_name_AD = dataset_name

    # MODEL #################################
    # neuron
    if n:
        n = n
    else:
        n = 400  # other tested so far: 600, 800
    tau = 0.05
    model_input_dim = 3
    model_output_dim = 2

    # fb
    fb_density = fb_density
    fb_delay = fb_delay

    # VR perturbation
    rot_phi = rot_phi / 180 * np.pi

    # regularization
    alpha1 = 1e-3  # reg inp & out & fb
    gamma1 = 1e-3  # reg rec
    beta1 = 2e-3  # regularization on activity

    # clip gradients
    clipgrad = 0.2

    # learning rate & batch size
    lr = learning_rate
    batch_size = batch_size

    # PROTOCOL ###############################
    if protocol is None:
        protocol = [["random_pushed", 250]]
    else:
        protocol = protocol
    # DATA #################################
    ntrials = 1000
    tsteps = 125
    dt = 0.01
    dataset_name = dataset_name
    p_test = 0.1  # test set size

    vel = vel
    go_to_peak = go_to_peak  # 50
    stim_on = 20

    if not custom_delay:
        r_go_range = [70, 220]
        cor_go_range = [170, 220]
    else:
        r_go_range = custom_delay["r_go_range"]
        cor_go_range = custom_delay["cor_go_range"]

    # random reach data set
    r_output_range = [-6, 6]
    # center out reach data set
    cor_output_range = 5
    ntargets = 8

    # Sinewave ##########
    sinewave = {
        "input_dim": 1,
        "output_dim": 2,
        "amplitude": 1,
        "freq_range": [freq,freq+3] if freq else [1, 4],
    }
    # Add ##########
    add = {
        "input_dim": 1,
        "output_dim": control_dim,
        "n-back": [2, 5],
        "time_window": 20,
    }

    # RANDOM PUSH PERTURBATION ###############
    p1_amp = 10
    p1_pratio = pratio # 0.25
    p1_halflength = 5
    p1_from = 20
    p1_upto = 190

    # SAVE IT ALL ##############################

    training = {
        "wfb_frozen_phase": wfb_frozen_phase,
    }

    model = {
        # neuron
        "n": n,
        "tau": tau,
        "input_dim": model_input_dim,
        "output_dim": model_output_dim,
        # time and reproducability
        "dt": dt,
        "tsteps": tsteps,
        "rand_seed": rand_seed,
        # fb
        "fb_density": fb_density,
        "fb_delay": fb_delay,
        # simulation protocol
        "protocol": protocol,
        "rot_phi": rot_phi,
        "freqshift": freqshift,
        # ml regularization
        "alpha1": alpha1,
        "beta1": beta1,
        "gamma1": gamma1,
        "clipgrad": clipgrad,
        # ml training
        "lr": lr,
        "batch_size": batch_size,
        "record_gradients": record_gradients,
        "get_grads_per_example": get_grads_per_example,
        "record_jacobians": record_jacobians,
        "record_weights": record_weights,
        "record_lca": record_lca,
        "record_weights_norm": record_weights_norm,
        "record_r_inputs": record_r_inputs,
        "recurrent": recurrent,
        "error_type": error_type,
        "error_detach": error_detach,
        "nlayers": nlayers,
        "fb_freeze": fb_freeze,
    }

    data = {
        "ntrials": ntrials,
        "tsteps": tsteps,
        "dt": dt,
        "dataset_name": dataset_name,
        "dataset_name_AD": dataset_name_AD,
        "p_test": p_test,
        "Reaching": {
            "input_dim": 7,
            "output_dim": 4,
            "vel": vel,
            "p_test": p_test,
            "go_to_peak": go_to_peak,
            "stim_on": stim_on,
            "random": {"output_range": r_output_range, "go_range": r_go_range},
            "center-out-reach": {
                "output_range": cor_output_range,
                "ntargets": ntargets,
                "go_range": cor_go_range,
            },
        },
        "Sinewave": sinewave,
        "Add": add,
    }

    p1 = {
        "amp": p1_amp,
        "pratio": p1_pratio,
        "halflength": p1_halflength,
        "from": p1_from,
        "upto": p1_upto,
    }

    params = {"model": model, "data": data, "savname": savname, "p1": p1, "training": training}
    np.save(savname + "/params", params)
    shutil.copy(__file__, savname + "/" + __file__.split("/")[-1])
    return params


if __name__ == "__main__":
    main()
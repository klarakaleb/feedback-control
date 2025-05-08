#%%

"""
Load initially trained network model and run adaptation algorithm.

    - biologically plausible learning rule
    - gradient descent (backprop)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, os, copy
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict

import sys

sys.path.append("modules")

from model_def import RNN
from data_set import *
from toolbox import *
from utils import normalized_dot_product
from learn_alg import *

import os

#%%


def run_single_model(
    model0,
    params,
    learning_rule,
    learning_rate,
    savname,
    fb_in,
    bp_opt=None,
    use_transpose=False,
    compute_hessian=False,
):
    # first make copy of initially trained model
    model = copy.deepcopy(model0)

    # set all except recurrent weights static
    model.output.weight.requires_grad = False
    model.output.bias.requires_grad = False
    model.rnn.weight_ih_l0.requires_grad = False
    model.feedback.weight.requires_grad = False
    model.feedback.bias.requires_grad = False

    # SETUP LOSS ######################
    criterion = nn.MSELoss(reduction="none")

    if learning_rule == "bp" or learning_rule == "BP":
        optimizer = BPTT(lr=learning_rate)
    elif learning_rule == 'sam' or learning_rule == 'SAM':
        optimizer = SAM(lr=learning_rate)
    elif learning_rule == 'bp_2' or learning_rule == 'BP_2':
        optimizer = BPTT_2(lr=learning_rate)
    elif learning_rule == "rflo" or learning_rule == "RFLO":
        optimizer = RFLO(lr=learning_rate)
    elif learning_rule == "fed" or learning_rule == "FED":
        optimizer = FED(lr=learning_rate)
    elif learning_rule == "rflo_t" or learning_rule == "RFLO_t":
        optimizer = RFLO_t(lr=learning_rate)
    elif learning_rule == "fed_t" or learning_rule == "FED_t":
        optimizer = FED_t(lr=learning_rate)
    else:
        print("Learning rule not implemented!!")

    if params["model"]["fb_density"] == 0 and (
        learning_rule in ["rflo", "RFLO", "fed", "FED"]
    ):
        # set fb weight to transpose (just for simplicity for now)
        state_dict = model.state_dict()
        state_dict["feedback.weight"] = model.output.weight.T
        model.load_state_dict(state_dict)

    # SET UP DATA ######################
    if params["data"]["dataset_name_AD"] == "Reaching":
        dataset = Reaching()
        pert_name = "center-out-reach_rotated"
    elif params["data"]["dataset_name_AD"] == "Sinewave":
        dataset = Sinewave()
        pert_name = "sine_freqshift"
    elif params["data"]["dataset_name_AD"] == "Add":
        dataset = Add()
        pert_name = "add_nbackshift"

    lc = []  # losses
    lcT = []  # losses test
    r_inputs = []
    outputs = []
    raw_outputs = []
    dws = []
    
    # comparions with optimal (BPTT)
    local_alignments = []
    local_alignments_2 = []
    hessian_eigvals = []

    # learning efficiency
    update_cost_stepwise = []
    update_cost = []
    update_alignment = []
    local_grads = []

    # genarate a test dataset featuring all 8 possible targets
    test_target, test_stimuli, test_perts, test_stim_ref = [], [], [], []
    if params["data"]["dataset_name_AD"] == "Reaching":
        while len(test_target)<8: # hacky way to get all 8 unique targets
            # sample
            target, stimulus, pert, tids, stim_ref = dataset.prepare_pytorch(
                params, pert_name, ntrials=1, batch_size=1
            )
            if not any(torch.equal(target, t) for t in test_target):
                test_target.append(target)
                test_stimuli.append(stimulus)
                test_perts.append(pert)
                test_stim_ref.append(stim_ref)
        print('generated test dataset with all 8 unique targets.')

        # stack + reshape accordingly
        test_target = torch.swapaxes(torch.cat(test_target),0,2)
        test_stimuli = torch.swapaxes(torch.cat(test_stimuli),0,2)
        test_perts = torch.swapaxes(torch.cat(test_perts),0,2)
        test_stim_ref = torch.swapaxes(torch.cat(test_stim_ref),0,2)

    elif params["data"]["dataset_name_AD"] == "Sinewave":
        test_target, test_stimuli, test_perts, _, test_stim_ref = dataset.prepare_pytorch(
            params, pert_name, ntrials=1, batch_size=100
        )


    criterion = torch.nn.MSELoss()

    if not learning_rule == "ctrl":  # i.e. some learning is happening
        # START ADAPTATION #################
        model.train()
        for epoch in range(params["ntrials"]):
            # create some data
            target, stimulus, pert, tids, stim_ref = dataset.prepare_pytorch(
                params, pert_name, ntrials=1, batch_size=params["batch_size"]
            )
            toprint = OrderedDict()

            # check loss without fb
            output, hidden, extras = model(
                test_stimuli[0],
                test_perts[0],
                test_stim_ref[0],
                fb_in=False,
            )
            lcT.append(criterion(output, output * 0).mean().detach().item())

            # now onto local learning
            init_w = model.rnn.weight_hh_l0.detach().clone()
            output, hidden, extras = model(
                stimulus[0],
                pert[0],
                stim_ref[0],
                optimizer,
                fb_in=fb_in,
                use_transpose=use_transpose,
                compute_hessian=compute_hessian,
            )
            final_w = model.rnn.weight_hh_l0.detach().clone()
            assert not torch.equal(init_w,final_w)
            dw = final_w - init_w

            loss_train = criterion(output, output * 0).mean()

            if epoch <= 2000:
                hessian_eigvals.append(extras["hessian_eigvals"])
                local_alignments.append(extras["local_alignments"])
                local_alignments_2.append(extras["local_alignments_2"])

                l_grads = torch.stack(extras["local_grads"]).detach()
                if epoch == 0 and params["ntrials"]<=1000:
                    local_grads.append(l_grads)

                update_cost.append(torch.linalg.norm(dw.detach()).detach())
                update_cost_stepwise_i = torch.sum(torch.stack([torch.linalg.norm(g.detach()) for g in l_grads]))
                update_cost_stepwise.append(update_cost_stepwise_i.detach())

                # record dws per epoch 
                if params["ntrials"]<=1000:
                    dws.append(dw.detach().clone())


            toprint["Loss"] = loss_train

            if epoch % 10 == 0:
                print(
                    ("Epoch=%d | " % (epoch))
                    + " | ".join("%s=%.4f" % (k, v) for k, v in toprint.items())
                )
            lc.append(loss_train.detach().item())

            if epoch % 100 == 0:

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "params": params,
                    },
                    savname + learning_rule + f"_{epoch}",
                )

                # test model
                with torch.no_grad():
                    loss_test = dataset.test_ad(
                        model=model, model_name=learning_rule, lc=lc, savname=savname, fb_in=0, params=params, pert_name=pert_name
                    )

            if learning_rule =='bp_2': 
                print('saving progress for epoch:', epoch)

                torch.save(
                {
                    "update_cost": update_cost,
                    "update_cost_stepwise": update_cost_stepwise,
                    "update_alignment": update_alignment,
                    "local_alignments": local_alignments,
                    "local_alignments_2": local_alignments_2,
                    "hessian_eigvals": hessian_eigvals,
                    "dws": dws,
                },
                    savname + learning_rule + "_grads_" + f"{epoch}",
                )

        print("MODEL TRAINED!")

        # check loss without fb
        output, hidden, extras = model(
            test_stimuli[0],
            test_perts[0],
            test_stim_ref[0],
            fb_in=False,
        )
        lcT.append(criterion(output, output * 0).mean().detach().item())

    # test model
    with torch.no_grad():
        loss_test = dataset.test_ad(
            model=model, model_name=learning_rule, lc=lc, savname=savname, fb_in=0, params=params, pert_name=pert_name
        )

    # save everything
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lc": np.array(lc),
            "lcT": np.array(lcT) if lcT else None,
            "loss_test": loss_test.detach(),
            "params": params,
        },
        savname + learning_rule,
    )

    torch.save(
        {
            "r_inputs": r_inputs,
            "output_e": outputs,  # save outputs with learning
            "output_raw": raw_outputs,  # save outputs with learning
            "target": stim_ref[0].detach(),  # save the constant target
        },
        savname + learning_rule + "_io",
    )

    torch.save(
        {
            "update_cost": update_cost,
            "update_cost_stepwise": update_cost_stepwise,
            "update_alignment": update_alignment,
            "local_alignments": local_alignments,
            "local_alignments_2": local_alignments_2,
            "hessian_eigvals": hessian_eigvals,
            "dws": dws,
            "local_grads": local_grads,
        },
        savname + learning_rule + "_grads",
    )


def main(
    savname,
    learning_rule,
    learning_rate,
    fb_in=True,
    tm_savname=None,
    use_transpose=False,
    ntrials=500,
    bp_opt="sgd",
    rot_phi=30,
    freqshift=1,
    batch_size=1,
    compute_grad=True,
    compute_hessian=False,
):
    
    print(freqshift)

    if not tm_savname:
        tm_savname = savname

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataT = torch.load(tm_savname + "/phase0_training", map_location=device)
    params = dataT["params"]

    savname += "/AD_"

    # ADAPTATION PARAMETERS #############
    batch_size = batch_size
    ntrials = ntrials
    ad_protocol = {learning_rule: learning_rate}
    rot_phi = rot_phi / 180 * np.pi


    params["model"].update({"rot_phi": rot_phi})
    params["model"].update({"freqshift": freqshift})

    print("params:", params["model"]["freqshift"])
                          
    keys = list(ad_protocol.keys())
    # SETUP SIMULATION #################
    rand_seed = params["model"]["rand_seed"]
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # GPU usage
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # SETUP MODEL #################
    model = RNN(
        params["model"]["input_dim"],
        params["model"]["output_dim"],
        params["model"]["n"],
        dtype,
        params["model"]["dt"],
        params["model"]["tau"],
        fb_delay=params["model"]["fb_delay"],
        fb_density=params["model"]["fb_density"],
        error_detach=True
    )
    if dtype == torch.cuda.FloatTensor:
        model = model.cuda()
    model.load_state_dict(dataT["model_state_dict"])
    print(model.rnn.weight_hh_l0.shape)

    w_ratio = model.feedback.weight.norm()/model.output.weight.norm()

    if learning_rule not in ['fed','rflo']:
        updated_learning_rate = w_ratio * learning_rate
        ad_protocol[learning_rule] = updated_learning_rate
    
    params.update(
        {"ad_protocol": ad_protocol, "ntrials": ntrials, "batch_size": batch_size}
    )

    if params["data"]["dataset_name_AD"] == "Reaching":
        model.pos_err = True

    for j in range(len(keys)):
        run_single_model(
            model,
            params,
            keys[j],
            ad_protocol[keys[j]],
            savname,
            fb_in=fb_in,
            bp_opt=bp_opt,
            use_transpose=use_transpose,
            compute_hessian=compute_hessian,
        )
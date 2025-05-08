%%
"""
Run whole simulation for network model.

    - initial training on random reach data (following what's defined in protocol')
    - biologically plausible adaptation run
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict
import itertools

import sys
import copy

from modules.utils import get_nullspace_relative_norm, get_jacobian, get_jacobian_2

sys.path.append("modules")

from data_set import *
from model_def import RNN
from utils import *


def main(savname):

    savname = savname + "/"

    params = np.load(savname + "params.npy", allow_pickle=True).item()
    protocol = params["model"]["protocol"]

    # SETUP SIMULATION #################
    rand_seed = params["model"]["rand_seed"]
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # GPU usage #################
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["model"].update({"dtype": dtype, "device": device})

    # DATASET #################
    if params["data"]["dataset_name"] == "Reaching":
        dataset = Reaching()
    elif params["data"]["dataset_name"] == "Sinewave":
        dataset = Sinewave()
        params["model"]["input_dim"] = params["data"]["Sinewave"]["input_dim"]
        params["model"]["output_dim"] = params["data"]["Sinewave"]["output_dim"]
    elif params["data"]["dataset_name"] == "Add":
        dataset = Add()
        params["model"]["input_dim"] = params["data"]["Add"]["input_dim"]
        params["model"]["output_dim"] = params["data"]["Add"]["output_dim"]
    elif params["data"]["dataset_name"] == "Add_zc":
        dataset = Add_zc()
        params["model"]["input_dim"] = params["data"]["Add"]["input_dim"]
        params["model"]["output_dim"] = params["data"]["Add"]["output_dim"]

    # SETUP MODEL #################

    if params["model"]["nlayers"] == 1:
        model = RNN(
            params["model"]["input_dim"],
            params["model"]["output_dim"],
            params["model"]["n"],
            dtype,
            params["model"]["dt"],
            params["model"]["tau"],
            fb_delay=params["model"]["fb_delay"],
            fb_density=params["model"]["fb_density"],
            recurrent=params["model"]["recurrent"],
            error_type=params["model"]["error_type"],
            error_detach=params["model"]["error_detach"],
        )
        init_state = copy.deepcopy(list(model.parameters()))
        print(model.rnn.weight_ih_l0.shape)
    elif params["model"]["nlayers"] == 2:
        model = RNN_2(
            params["model"]["input_dim"],
            params["model"]["output_dim"],
            params["model"]["n"],
            dtype,
            params["model"]["dt"],
            params["model"]["tau"],
            fb_delay=params["model"]["fb_delay"],
            fb_density=params["model"]["fb_density"],
            recurrent=params["model"]["recurrent"],
            error_type=params["model"]["error_type"],
            error_detach=params["model"]["error_detach"],
        )

    # TO CUDA OR NOT TO CUDA #################
    if dtype == torch.cuda.FloatTensor:
        model = model.cuda()

    # TASK STUFF #################
    if params["data"]["dataset_name"] == "Reaching":
        model.pos_err = True

    # SETUP OPTIMIZER #################
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=params["model"]["lr"]
    )

    get_grads_per_example = params["model"]["get_grads_per_example"]

    # START INITIAL (PRE)TRAINING #################
    for phase in range(len(protocol)):
        print("\n####### PHASE %d #######" % phase)
        # get info from protocol
        ph_ntrials = protocol[phase][1]
        ph_datname = protocol[phase][0]
        batch_size = params["model"]["batch_size"]

        # test task
        if params["data"]["dataset_name"] == "Reaching":
            test_task = "center-out-reach"
            test_perturbation = "_rotated"
        elif params["data"]["dataset_name"] == "Sinewave":
            test_task = "sinewave"
            test_perturbation = "_shifted"
        else:
            print('what are you trying to do?')

        # create test_1 dataset to eval on
        target_t, stimulus_t, pert_t, tids_t, stim_ref_t = dataset.prepare_pytorch(
            params, test_task, test_set=True
        )
        
        # create test_2 dataset to eval on (with a pertubation)
        target_tp, stimulus_tp, pert_tp, tids_tp, stim_ref_tp = dataset.prepare_pytorch(
            params, test_task +  test_perturbation, test_set=True
        )  

        target_tr, stimulus_tr, pert_tr, tids_tr, stim_ref_tr = dataset.prepare_pytorch(
            params, 'random', test_set=True
        )

        target_trp, stimulus_trp, pert_trp, tids_trp, stim_ref_trp = dataset.prepare_pytorch(
            params, 'random_pushed', test_set=True
        )

        # ACTUAL TRAINING STARTS
        lc = []
        model.train()

        # PREPARE TO RECORD #########

        lca = defaultdict(list)
        delta_w_norms = defaultdict(list)

        batch_var_ins = []
        batch_var_fbs = []
        output_vars = []

        wfbs_a = []
        alignments = []
        coherences = []
        norms = []
        raw_outputs = []

        test_losses = []
        test_losses_p = []
        test_losses_r = []
        test_losses_r_nofb = []
        test_losses_rp = []
        pt_alignments = []
        pt_alignments_2 = []
        nrns = []

        for epoch in range(ph_ntrials):
            # create data for this training phase

            if epoch < params["training"]["wfb_frozen_phase"]:
                model.feedback.weight.requires_grad = False
            else:
                model.feedback.weight.requires_grad = True

            if params["training"]["wfb_frozen_phase"] == 0:
                target, stimulus, pert, tids, stim_ref = dataset.prepare_pytorch(
                    params, ph_datname, 1, batch_size
                )
            else:
                if epoch < 500 + params["training"]["wfb_frozen_phase"]:
                    target, stimulus, pert, tids, stim_ref = dataset.prepare_pytorch(
                        params, ph_datname, 1, batch_size
                    )
                else:
                    target, stimulus, pert, tids, stim_ref = dataset.prepare_pytorch(
                        params, "random_pushed", 1, batch_size
                    )   


            # get batch example variance
            if epoch % 10 == 0:
                batch_var = stimulus[0].var(dim=1)
                batch_var_ins.append(batch_var)

            if (params["model"]["fb_freeze"] and epoch < 100) or params["model"][
                "fb_density"
            ] == 0:
                model.feedback.weight.requires_grad = False
            else:
                model.feedback.weight.requires_grad = True

            # PREP WORK #######
            optimizer.zero_grad()
            loss = torch.tensor(0.0).to(model.rnn.weight_ih_l0.device)
            toprint = OrderedDict()

            with torch.no_grad():
                if params["data"]["dataset_name"] == "Reaching":
                    loss_test = dataset.test_dt(model, stimulus_t, pert_t, stim_ref_t, fb_in=True)
                    test_losses.append(loss_test.detach())
                    loss_test = dataset.test_dt(model, stimulus_tp, pert_tp, stim_ref_tp, fb_in=True)
                    test_losses_p.append(loss_test.detach())
                loss_test = dataset.test_dt(model, stimulus_tr, pert_tr, stim_ref_tr, fb_in=True)
                test_losses_r.append(loss_test.detach())
                loss_test = dataset.test_dt(model, stimulus_tr, pert_tr, stim_ref_tr, fb_in=False)
                test_losses_r_nofb.append(loss_test.detach())
                loss_test = dataset.test_dt(model, stimulus_trp, pert_trp, stim_ref_trp, fb_in=True)
                test_losses_rp.append(loss_test.detach())


            # add regularization
            # term 1: parameters
            regin = params["model"]["alpha1"] * model.rnn.weight_ih_l0.norm(2)
            regout = params["model"]["alpha1"] * model.output.weight.norm(2)
            regoutb = params["model"]["alpha1"] * model.output.bias.norm(2)
            regfb = params["model"]["alpha1"] * model.feedback.weight.norm(2)
            regfbb = params["model"]["alpha1"] * model.feedback.bias.norm(2)
            regrec = params["model"]["gamma1"] * model.rnn.weight_hh_l0.norm(2)
            reg = regin + regrec + regout + regoutb + regfbb + regfb

            all_per_sample_gradients = []
            if get_grads_per_example and epoch%10==0: # every tenth for efficiency
                for example in range(batch_size):
                    output, hidden, extras = model(
                        stimulus[0][:, example][:, None, :],
                        pert[0][:, example][:, None, :],
                        stim_ref[0][:, example][:, None, :],
                    )
                    loss_train = criterion(output, output * 0).mean()  # mean across t

                    # term 2: rates
                    regact = params["model"]["beta1"] * hidden.pow(2).mean()

                    loss += loss_train.item()

                    loss_total = (loss_train + reg + regact) / batch_size

                    loss_total.backward(retain_graph=True)

                    per_sample_gradients = [
                        p.grad.detach().clone()
                        for p in model.parameters()
                        if p.requires_grad
                    ]

                    all_per_sample_gradients.append(per_sample_gradients)
                    model.zero_grad()  # p.grad is cumulative so we'd better reset it

                loss = loss * 1 / batch_size
                loss_train = loss

                mean_per_param_gradients = [
                    sum(i) for i in zip(*all_per_sample_gradients)
                ]

                # calculate alignment of each sample with mean (true) gradient
                all_per_sample_alignments = []
                all_per_sample_norms = []
                for example in range(batch_size):
                    alignment = [
                        torch.nn.functional.cosine_similarity(
                            x.flatten(), y.flatten(), dim = 0
                        )
                        for x, y in zip(
                            all_per_sample_gradients[example],
                            mean_per_param_gradients,
                        )
                    ]
                    all_per_sample_alignments.append(alignment)

                    norm = [
                        torch.linalg.norm(x) for x in all_per_sample_gradients[example]
                    ]  # norm of each parameter gradient
                    
                    all_per_sample_norms.append(norm)

                # make the below such that it is all against all
                pairwise_dot = []
                for example_x, example_y in itertools.combinations(all_per_sample_gradients,2):
                    dot_product = [torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(),dim=0) for x, y in zip(example_x, example_y)] # per param
                    pairwise_dot.append(dot_product) 
                average_dot = torch.tensor(pairwise_dot).mean(dim=0)
                coherence = average_dot


                coherences.append(coherence)
                alignments.append(all_per_sample_alignments)
                norms.append(all_per_sample_norms)

                c = 0
                for i, p in enumerate(model.parameters()):
                    if p.requires_grad:
                        p.grad = mean_per_param_gradients[c]
                        c += 1

            else:
                # FORWARD PASS ############
                output, hidden, extras = model(
                    stimulus[0],
                    pert[0],
                    stim_ref[0],
                )


                # get batch example variance
                if epoch % 10 == 0:
                    batch_var = output.detach().var(dim=1)
                    batch_var_fbs.append(batch_var)
                    output_var = extras["raw_output"].detach().var(dim=1)
                    output_vars.append(output_var)

                # BACKWARD PASS ############
                loss_train = criterion(output, output * 0).mean()
                toprint["Loss"] = loss_train
                regact = params["model"]["beta1"] * hidden.pow(2).mean()
                reg = regin + regrec + regout + regoutb + regfbb + regfb
                loss = loss_train + reg + regact
                loss.backward(retain_graph=True)

            # CLIP THOSE GRADIENTS TO AVOID EXPLOSIONS ########
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params["model"]["clipgrad"]
            )


            # for LCA
            if not get_grads_per_example:
                prev_w = defaultdict(list)
                grads = defaultdict(list)
                for i, p in enumerate(model.parameters()):
                    if p.requires_grad:
                        prev_w[i] = p.detach().clone()
                        grads[i] = p.grad.detach().clone()

                # record alignment of weights
                a = model.feedback.weight
                b = model.output.weight @ model.rnn.weight_hh_l0
                wfbs_e = torch.nn.functional.cosine_similarity(
                    a.detach().clone().flatten(),
                    b.T.detach().clone().flatten(),
                    dim=0,
                )
                a = model.feedback.weight
                b = model.output.weight @ model.rnn.weight_hh_l0
                wfbs_e2 = torch.nn.functional.cosine_similarity(
                    a.detach().clone().flatten(),
                    b.T.detach().clone().flatten(),
                    dim=0,)
                wfbs_a.append(torch.stack([wfbs_e, wfbs_e2]))

            # APPLY GRADIENTS TO PARAMETERS ########
            optimizer.step()

            if not get_grads_per_example:
                for i, p in enumerate(model.parameters()):
                    if p.requires_grad:
                        delta_w = p.detach().clone() - prev_w[i]
                        delta_w_norms[i].append(torch.linalg.norm(delta_w))
                        lca_i = torch.dot(delta_w.flatten(), grads[i].flatten())
                        lca[i].append(lca_i)

            train_running_loss = [
                loss_train.detach().item(),
                regact.detach().item(),
                regin.detach().item(),
                regrec.detach().item(),
                regout.detach().item(),
                regoutb.detach().item(),
                regfb.detach().item(),
                regfbb.detach().item(),
            ]
            # printing
            toprint["Loss"] = loss
            toprint["In"] = regin
            toprint["Rec"] = regrec
            toprint["Out"] = regout
            toprint["OutB"] = regoutb
            toprint["Fb"] = regfb
            toprint["FbB"] = regfbb
            toprint["Act"] = regact

            print(
                ("Epoch=%d | " % (epoch))
                + " | ".join("%s=%.4f" % (k, v) for k, v in toprint.items())
            )
            lc.append(train_running_loss)

            if epoch % 100 == 0:
                torch.save(
                    {
                        "epoch": ph_ntrials,
                        "model_state_dict": model.state_dict(),
                        "model_init_state_dict": init_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lc": np.array(lc),
                        "loss_test": test_losses,  
                        "loss_test_p": test_losses_p,  
                        "loss_test_r": test_losses_r,
                        "loss_test_r_nofb": test_losses_r_nofb,
                        "loss_test_rp": test_losses_rp,
                        "params": params,
                    },
                    savname + "phase" + str(phase) + "_training_" + str(epoch),
                )

        print("MODEL TRAINED!")
        with torch.no_grad():
            loss_test = dataset.test(model, lc, params, ph_datname, savname, phase)
            # test_losses.append(loss_test.detach())
        print("MODEL TESTED!")
        # save this phase
        torch.save(
            {
                "epoch": ph_ntrials,
                "model_state_dict": model.state_dict(),
                "model_init_state_dict": init_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "lc": np.array(lc),
                "loss_test": test_losses,  
                "loss_test_p": test_losses_p,  
                "loss_test_r": test_losses_r,
                "loss_test_r_nofb": test_losses_r_nofb,
                "loss_test_rp": test_losses_rp,
                "params": params,
            },
            savname + "phase" + str(phase) + "_training",
        )
        torch.save(
            {
                "output_raw": raw_outputs,  # save outputs with learning
                "target": stim_ref[0].detach(),  # save the constant target
            },
            savname + "phase" + str(phase) + "_io",
        )

        torch.save(
            pt_alignments,
            savname + "phase" + str(phase) + "_c_align",
        )

        torch.save(
            pt_alignments_2,
            savname + "phase" + str(phase) + "_a_align",
        )

        torch.save(
            alignments,
            savname + "phase" + str(phase) + "_alignments",
        )
        torch.save(
            coherences,
            savname + "phase" + str(phase) + "_coherences",
        )
        torch.save(
            norms,
            savname + "phase" + str(phase) + "_norms",
        )
        torch.save(
            wfbs_a,
            savname + "phase" + str(phase) + "_wfbs_a",
        )
        torch.save(
            lca,
            savname + "phase" + str(phase) + "_lca",
        )
        torch.save(
            lca,
            savname + "phase" + str(phase) + "_lca",
        )
        torch.save(
            delta_w_norms,
            savname + "phase" + str(phase) + "_delta_w_norms",
        )
        torch.save(
            batch_var_ins,
            savname + "phase" + str(phase) + "_batch_var_ins",
        )
        torch.save(
            batch_var_fbs,
            savname + "phase" + str(phase) + "_batch_var_fbs",
        )
        torch.save(
            output_vars,
            savname + "phase" + str(phase) + "_output_vars",
        )



if __name__ == "__main__":
    main()
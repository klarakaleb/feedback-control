"""
Toolbox for data creation.
"""

import numpy as np
import torch
from toolbox import *
import matplotlib.pyplot as plt


class Task:
    def test(self, *argparams, **kwparams):
        pass

    def test_ad(self, *argparams, **kwparams):
        pass

    def prepare_pytorch(self, *argparams, **kwparams):
        pass

    def _perturb(self, *argparams, **kwparams):
        pass


class Reaching(Task):
    def test(self, model, lc, params, ph_datname, savname, phase):
        model.eval()
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, ph_datname, test_set=True
        )
        output, _, _ = model(stimulus, pert, stim_ref)
        loss_test = torch.mean(output ** 2)
        # create loss figure
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.plot(lc, "k")
        plt.ylabel("Train loss")
        plt.axhline(
            loss_test.detach().cpu(), linestyle="--", color="k", label="Test loss"
        )
        plt.legend(loc="upper right", frameon=False)
        plt.title(ph_datname, fontsize=10)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.semilogy()
        plt.subplot(1, 3, 2)
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, "center-out-reach", test_set=True
        )
        _, hidden, _ = model(stimulus, pert, stim_ref)
        plot_trajectories(
            model.get_output(hidden).cpu().detach().numpy().transpose(1, 0, 2),
            tids,
            target,
        )
        plt.text(-8, -8, "center-out-reach", fontsize=8)
        plt.subplot(1, 3, 3)
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, "center-out-reach_rotated", test_set=True
        )
        _, hidden, _ = model(stimulus, pert, stim_ref)
        plot_trajectories(
            model.get_output(hidden).cpu().detach().numpy().transpose(1, 0, 2),
            tids,
            target,
        )
        plt.text(-8, -8, "center-out-reach_rotated", fontsize=8)
        plt.suptitle(savname.split("/")[-1])
        plt.savefig(
            savname + "phase" + str(phase) + "_loss.png", bbox_inches="tight", dpi=150
        )
        plt.savefig(savname + "phase" + str(phase) + "_loss.svg", bbox_inches="tight")
        plt.close()

        return loss_test

    def test_dt(self, model, stimulus, pert, stim_ref, fb_in):
        model.eval()
        output, hidden, _ = model(X = stimulus, Xpert = pert, Xref = stim_ref, fb_in = fb_in)
        loss_test = torch.mean(output ** 2)
        output = model.get_output(hidden)
        return loss_test

    def test_ad(self, model, model_name, lc, savname, fb_in, params, pert_name):
        model.eval()
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, pert_name, test_set=True
        )
        output, hidden, _ = model(stimulus, pert, stim_ref, fb_in=fb_in)
        loss_test = torch.mean(output ** 2)
        # create loss figure
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.ylabel("Train loss")
        plt.plot(lc, "k")
        plt.axhline(
            loss_test.detach().cpu(), linestyle="--", color="k", label="Test loss"
        )
        plt.legend(loc="upper right", frameon=False)
        plt.title(pert_name, fontsize=10)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.semilogy()
        plt.subplot(1, 3, 2)
        plot_trajectories(
            model.get_output(hidden).cpu().detach().numpy().transpose(1, 0, 2),
            tids,
            target,
        )
        plt.text(-8, -8, "center-out-reach_rotated", fontsize=8)
        plt.subplot(1, 3, 3)
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, "center-out-reach", test_set=True
        )
        output, hidden, _ = model(stimulus, pert, stim_ref, fb_in=fb_in)
        plot_trajectories(
            model.get_output(hidden).cpu().detach().numpy().transpose(1, 0, 2),
            tids,
            target,
        )
        plt.text(-8, -8, "center-out-reach", fontsize=8)
        plt.suptitle(model_name, y=1.1)
        plt.savefig(savname + model_name + ".png", bbox_inches="tight", dpi=150)
        plt.savefig(savname + model_name + ".svg", bbox_inches="tight")
        plt.close()
        print("MODEL " + model_name + " TESTED!")
        # save this phase
        return loss_test

    def prepare_pytorch(
        self,
        params,
        dataset_name,
        ntrials=100,
        batch_size=1,
        test_set=False,
        rot_phi=None,
    ):
        dname = dataset_name.split("_")[0]
        pertname = (
            dataset_name.split("_")[1] if len(dataset_name.split("_")) > 1 else ""
        )

        # construct artifical data
        if dname == "center-out-reach":
            data = self._create_data_velocity_centeroutreach(params["data"])
        elif dname == "random":
            data = self._create_data_velocity_random(params["data"])

        if test_set:
            data = data["test_set"]

        # dtype = params["model"]["dtype"]
            
        # GPU usage
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
            
        # GPU usage
        tout = data["target"][:, :, 2:]
        tstim = data["stimulus"]

        # apply perturbation
        if pertname == "rotated":
            rot_phi = params["model"]["rot_phi"]
            rotmat = np.array(
                [
                    [np.cos(rot_phi), -np.sin(rot_phi)],
                    [np.sin(rot_phi), np.cos(rot_phi)],
                ]
            )
            tout = tout @ rotmat.T
            tstim[:, :, 3:5] = tstim[:, :, 3:5] @ rotmat.T

        if test_set:
            target = tout
            stimulus = torch.Tensor(tstim.transpose(1, 0, 2)).type(dtype)
            pert = torch.zeros(tout.transpose(1, 0, 2).shape).type(dtype)
            try:
                tids = data["tids"]
            except:
                tids = None
            return target, stimulus[:, :, :3], pert, tids, stimulus[:, :, 3:5]
        else:
            target, pert, stimulus = [], [], []
            tids = []
            for j in range(ntrials):
                idx = np.random.choice(range(tout.shape[0]), batch_size, replace=False)
                tids.append(idx)
                target.append(torch.Tensor(tout[idx].transpose(1, 0, 2)))
                stimulus.append(torch.Tensor(tstim[idx].transpose(1, 0, 2)))
                # insert random push perturbation
                if pertname == "pushed":
                    p = torch.zeros(
                        params["model"]["tsteps"],
                        batch_size,
                        params["model"]["output_dim"],
                    )
                    pert.append(self._perturb(p, batch_size, params["p1"]))
                else:
                    pert.append(
                        torch.zeros(
                            size=[
                                params["model"]["tsteps"],
                                batch_size,
                                params["model"]["output_dim"],
                            ]
                        )
                    )

            target = torch.stack(target) if len(target) > 1 else target[0][None, :]
            stimulus = (
                torch.stack(stimulus) if len(stimulus) > 1 else stimulus[0][None, :]
            )
            pert = torch.stack(pert) if len(pert) > 1 else pert[0][None, :]
            return (
                target,
                stimulus[:, :, :, :3],
                pert,
                tids,
                torch.cat(
                    [
                        torch.zeros_like(stimulus[:, :1, :, 3:5]),
                        stimulus[:, :-1, :, 3:5],
                    ],
                    dim=1,
                ),
            )

    def _prepare_data(
        self,
        start_point,
        end_point,
        go_on,
        vel,
        tsteps,
        input_dim,
        output_dim,
        dt,
        stim_range,
        go_to_peak,
        stim_on,
    ):
        ntrials = start_point.shape[0]

        def sig(x, beta):
            return 1 / (1 + np.exp(-x * beta))

        # prepare xaxis for smooth transition
        xx = np.linspace(-1, 1, 100, endpoint=False)
        ytemp = sig(xx, vel)

        # create target
        target = np.zeros((ntrials, tsteps, output_dim))
        for j in range(ntrials):
            target[j, : (go_on[j] + go_to_peak), :2] = start_point[j]
            target[j, (go_on[j] + go_to_peak) :, :2] = end_point[j]
            target[j, (go_on[j] - go_to_peak) : (go_on[j] + go_to_peak), :2] += (
                ytemp[:, None] * (end_point[j] - start_point[j])[None, :]
            )

        # add target velocity
        target[:, :, 2:] = np.gradient(target[:, :, :2], dt, axis=1)

        # create stimulus
        stimulus = np.zeros((ntrials, tsteps, input_dim))
        stimulus[:, :, 3:5] = target[:, :, 2:]
        stimulus[:, :, 5:] = target[:, :, :2]
        for j in range(ntrials):
            stimulus[j, stim_on:, :2] = end_point[j] - start_point[j]
            stimulus[j, : (go_on[j] - go_to_peak), 2] = stim_range

        # # constant velocity
        # target = abs(target[:, :, :]).max(0) * (
        #     ((target[:, :, :] > 0).any(axis=0) * 2) - 1
        # )

        return target, stimulus

    def _create_data_velocity_random(self, data):
        # PARAMS #################################
        ntrials = data["ntrials"]
        tsteps = data["tsteps"]
        dt = data["dt"]
        output_dim = data["Reaching"]["output_dim"]
        input_dim = data["Reaching"]["input_dim"]
        vel = data["Reaching"]["vel"]
        p_test = data["Reaching"]["p_test"]
        go_to_peak = data["Reaching"]["go_to_peak"]
        stim_on = data["Reaching"]["stim_on"]
        output_range = data["Reaching"]["random"]["output_range"]
        go_range = data["Reaching"]["random"]["go_range"]
        ##########################################

        # create artifical data
        start_point = np.random.uniform(output_range[0], output_range[1], (ntrials, 2))
        end_point = np.random.uniform(output_range[0], output_range[1], (ntrials, 2))
        go_on = np.random.uniform(go_range[0], go_range[1], ntrials).astype(int)

        target, stimulus = self._prepare_data(
            start_point,
            end_point,
            go_on,
            vel,
            tsteps,
            input_dim,
            output_dim,
            dt,
            output_range[1],
            go_to_peak,
            stim_on,
        )

        # create testset
        test_idx = np.random.rand(ntrials) < p_test
        test_set = {
            "target": target[test_idx],
            "stimulus": stimulus[test_idx],
            "peak_speed": go_on[test_idx],
        }
        train_idx = test_idx == False

        # save it
        data = {
            "params": data,
            "target": target[train_idx],
            "peak_speed": go_on[train_idx],
            "stimulus": stimulus[train_idx],
            "test_set": test_set,
        }

        # print("RANDOM REACH DATASET CONSTRUCTED!")
        return data

    def _create_data_velocity_centeroutreach(self, data):
        # PARAMS #################################
        ntrials = data["ntrials"]
        tsteps = data["tsteps"]
        dt = data["dt"]
        output_dim = data["Reaching"]["output_dim"]
        input_dim = data["Reaching"]["input_dim"]
        vel = data["Reaching"]["vel"]
        p_test = data["Reaching"]["p_test"]
        go_to_peak = data["Reaching"]["go_to_peak"]
        stim_on = data["Reaching"]["stim_on"]
        output_range = data["Reaching"]["center-out-reach"]["output_range"]
        go_range = data["Reaching"]["center-out-reach"]["go_range"]
        ntargets = data["Reaching"]["center-out-reach"]["ntargets"]
        ##########################################

        # create artifical data
        start_point = np.zeros((ntrials, 2))
        phi = np.linspace(0, 2 * np.pi, ntargets, endpoint=False)
        tids = np.random.choice(range(ntargets), ntrials)
        end_point = (output_range * np.array([np.cos(phi[tids]), np.sin(phi[tids])])).T
        go_on = np.random.uniform(go_range[0], go_range[1], ntrials).astype(int)

        target, stimulus = self._prepare_data(
            start_point,
            end_point,
            go_on,
            vel,
            tsteps,
            input_dim,
            output_dim,
            dt,
            output_range,
            go_to_peak,
            stim_on,
        )

        # create testset
        test_idx = np.random.rand(ntrials) < p_test
        test_set = {
            "target": target[test_idx],
            "stimulus": stimulus[test_idx],
            "peak_speed": go_on[test_idx],
            "tids": tids[test_idx],
        }
        train_idx = test_idx == False

        # save it
        data = {
            "params": data,
            "target": target[train_idx],
            "peak_speed": go_on[train_idx],
            "stimulus": stimulus[train_idx],
            "test_set": test_set,
            "tids": tids[train_idx],
        }

        # print("CENTER OUT REACH DATASET CONSTRUCTED!")
        return data

    def _perturb(self, tdat, batch_size, dpert, dim=2):
        pratio = dpert["pratio"]
        halflength = dpert["halflength"]
        for i in range(batch_size):
            for l in range(dim):
                if np.random.rand() < pratio:
                    tmp = int(
                        np.random.choice(range(dpert["from"], dpert["upto"], 1), 1)[0]
                    )
                    tdat[(tmp - halflength) : (tmp + halflength), i, l] = dpert["amp"]
        return tdat


class Sinewave(Task):
    def prepare_pytorch(
        self, params, dataset_name, ntrials=100, batch_size=1, test_set=False, freqshift=None
    ):
        # get parameters
        amplitude = params["data"]["Sinewave"]["amplitude"]
        freq_range = params["data"]["Sinewave"]["freq_range"]
        dim = 2 #params["data"]["Sinewave"]["input_dim"]  # same as output dim
        n0trials = params["data"]["ntrials"]
        tsteps = params["data"]["tsteps"]
        p_test = params["data"]["p_test"]
        # dtype = params["model"]["dtype"]

        # GPU usage
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        dname = dataset_name.split("_")[0]
        pertname = (
            dataset_name.split("_")[1] if len(dataset_name.split("_")) > 1 else ""
        )

        # # construct artifical data
        # dim = int(dname)

        tout = np.zeros((n0trials, tsteps, dim))
        inputs = np.zeros((n0trials, tsteps, 1))
        freqs = np.random.uniform(freq_range[0], freq_range[1], (n0trials, dim))
        time = np.linspace(0, 2 * np.pi, tsteps, endpoint=False)

        # 1D
        warmup = 20 
        inputs[:, :, 0] = freqs[:, 0][:, None]
        inputs[:,:warmup,:] = 0 # some warmup period
        # sine AND cosine
        tout[:, warmup:, 0] = amplitude * np.sin(freqs[:, 0][:, None] * time[None, :-warmup])
        tout[:, warmup:, 1] = amplitude * np.cos(freqs[:, 0][:, None] * time[None, :-warmup])
        for j, freq in enumerate(freqs):
            tout[j, :warmup + int(0.99 + tsteps / (4 * freqs[j, 0])), 1] = 0
                
        test_idx = np.random.rand(n0trials) < p_test
        train_idx = test_idx == False

        # apply perturbation
        if pertname == "freqshift":
            freqshift = params["model"]['freqshift'] if freqshift is None else freqshift
            freqs += freqshift
            tout[:, warmup:, 0] = amplitude * np.sin(freqs[:, 0][:, None] * time[None, :-warmup])
            tout[:, warmup:, 1] = amplitude * np.cos(freqs[:, 0][:, None] * time[None, :-warmup])
            for j, freq in enumerate(freqs):
                tout[j, :warmup + int(0.99 + tsteps / (4 * freqs[j, 0])), 1] = 0

        if test_set:
            target = tout[test_idx]
            stimulus = torch.Tensor(inputs[test_idx].transpose(1, 0, 2)).type(dtype)
            pert = torch.zeros(target.transpose(1, 0, 2).shape).type(dtype)
            tids = None
            return (
                target,
                stimulus,
                pert,
                tids,
                torch.Tensor(target.transpose(1, 0, 2)).type(dtype),
            )
        else:
            tout = tout[train_idx]
            inputs = inputs[train_idx]

            target = torch.zeros(ntrials, tsteps, batch_size, dim).type(dtype)
            stimulus = torch.zeros(ntrials, tsteps, batch_size, 1).type(dtype)
            pert = torch.zeros(ntrials, tsteps, batch_size, dim).type(dtype)
            tids = []
            for j in range(ntrials):
                idx = np.random.choice(range(tout.shape[0]), batch_size, replace=False)
                tids.append(idx)
                target[j] = torch.Tensor(tout[idx].transpose(1, 0, 2)).type(dtype)
                stimulus[j] = torch.Tensor(inputs[idx].transpose(1, 0, 2)).type(dtype)
                # insert random push perturbation
                if pertname == "pushed":
                    pert[j] = self._perturb(
                        pert[j], batch_size, params["p1"], amp=amplitude / 4, dim=dim
                    )

            return target, stimulus, pert, tids, target

    def test(self, model, lc, params, ph_datname, savname, phase):
        model.eval()
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, ph_datname, test_set=True
        )
        output, hidden, _ = model(stimulus, pert, stim_ref)
        loss_test = torch.mean(output ** 2)
        output = model.get_output(hidden)
        # create loss figure
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.plot(lc, "k")
        plt.ylabel("Train loss")
        plt.axhline(
            loss_test.detach().cpu(), linestyle="--", color="k", label="Test loss"
        )
        plt.legend(loc="upper right", frameon=False)
        plt.title(ph_datname, fontsize=10)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.semilogy()

        plt.subplot(1, 3, 2)
        plt.plot(target[0, :, 0], "k")
        plt.plot(output[:, 0, 0].detach().cpu().numpy(), "r--")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.subplot(1, 3, 3)
        plt.plot(target[0, :, 1], "k")
        plt.plot(output[:, 0, 1].detach().cpu().numpy(), "r--")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.suptitle(savname.split("/")[-1])
        plt.savefig(
            savname + "phase" + str(phase) + "_loss.png", bbox_inches="tight", dpi=150
        )
        plt.savefig(savname + "phase" + str(phase) + "_loss.svg", bbox_inches="tight")
        plt.close()

        return loss_test

    def test_dt(self, model, stimulus, pert, stim_ref, fb_in):
        model.eval()
        output, _, _ = model(X = stimulus, Xpert = pert, Xref = stim_ref, fb_in = fb_in)
        loss_test = torch.mean(output ** 2)
        # output = model.get_output(hidden)
        return loss_test

    def test_ad(self, model, model_name, lc, savname, fb_in, params, pert_name):
        model.eval()
        target, stimulus, pert, tids, stim_ref = self.prepare_pytorch(
            params, pert_name, test_set=True
        )
        output, hidden, _ = model(X = stimulus, Xpert = pert, Xref = stim_ref, fb_in = fb_in)
        loss_test = torch.mean(output ** 2)
        output = model.get_output(hidden)
        # create loss figure
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.ylabel("Train loss")
        plt.plot(lc, "k")
        plt.axhline(
            loss_test.detach().cpu(), linestyle="--", color="k", label="Test loss"
        )
        plt.legend(loc="upper right", frameon=False)
        plt.title(pert_name, fontsize=10)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.semilogy()

        plt.subplot(1, 3, 2)
        plt.plot(target[0, :, 0], "k")
        plt.plot(output[:, 0, 0].detach().cpu().numpy(), "r--")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.subplot(1, 3, 3)
        plt.plot(target[0, :, 1], "k")
        plt.plot(output[:, 0, 1].detach().cpu().numpy(), "r--")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.suptitle(model_name, y=1.1)
        plt.savefig(savname + model_name + ".png", bbox_inches="tight", dpi=150)
        plt.savefig(savname + model_name + ".svg", bbox_inches="tight")
        plt.close()
        print("MODEL " + model_name + " TESTED!")
        # save this phase
        return loss_test

    def _perturb(self, tdat, batch_size, dpert, amp, dim=2):
        pratio = dpert["pratio"]
        halflength = dpert["halflength"]
        for i in range(batch_size):
            for l in range(dim):
                if np.random.rand() < pratio:
                    tmp = int(
                        np.random.choice(range(dpert["from"], dpert["upto"], 1), 1)[0]
                    )
                    tdat[(tmp - halflength) : (tmp + halflength), i, l] = amp
        return tdat



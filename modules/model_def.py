#!/usr/bin/env python3 3.7.4 project3 env
# -*- coding: utf-8 -*-
"""
Toolbox for model definition.
"""
from numpy import False_
import torch
import torch.nn as nn
from utils import get_jacobian, normalized_dot_product, relu_der, get_jacobian_2
from learn_alg import *


# random
class RNN(nn.Module):
    """Feedback of error: online position difference."""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_neurons,
        dtype,
        dt,
        tau,
        fb_delay=0,
        fb_density=1,
        pos_err=False,
        recurrent=True,
        error_type="error",
        error_detach=False,
    ):
        super(RNN, self).__init__()
        self.n_neurons = n_neurons
        self.alpha = dt / tau
        self.dt = dt

        self.rnn = nn.RNN(n_inputs, n_neurons, num_layers=1, bias=False)
        self.output = nn.Linear(n_neurons, n_outputs)
        self.feedback = nn.Linear(n_outputs, n_neurons)
        self.dtype = dtype

        self.nonlin = torch.nn.ReLU()
        self.nonlin_der = relu_der

        self.mask = nn.Linear(n_outputs, n_neurons, bias=False)
        self.mask.weight = nn.Parameter(
            (torch.rand(n_neurons, n_outputs) < fb_density).float()
        )
        self.mask.weight.requires_grad = False

        self.delay = fb_delay
        self.pos_err = pos_err

        self.recurrent = recurrent
        self.error_type = error_type
        self.error_detach = error_detach

    def init_hidden(self):
        return ((torch.rand(self.batch_size, self.n_neurons) - 0.5) * 0.2).type(
            self.dtype
        )

    # ONE SIMULATION STEP
    def f_step(self, xin, x1, r1, v1fb, v1, pin, xref, fb_in=True):
        in1 = xin.to(self.rnn.weight_ih_l0.T) @ self.rnn.weight_ih_l0.T
        in2 = r1 @ self.rnn.weight_hh_l0.T
        if not self.recurrent:
            in2 *= 0
        if self.error_detach:
            in3 = (
                v1fb.detach() @ (self.mask.weight * self.feedback.weight).T
            )  # feedback
            in3_s = v1fb @ (self.mask.weight * self.feedback.weight).T
        else:
            in3 = v1fb @ (self.mask.weight * self.feedback.weight).T
            in3_s = in3
        if fb_in:
            x1 = (1 - self.alpha) * x1 + self.alpha * (
                in1 + in2 + in3 + self.feedback.bias.T
            )
        else:
            x1 = (1 - self.alpha) * x1 + self.alpha * (in1 + in2 + self.feedback.bias.T)
        r1 = self.nonlin(x1)  # activation
        vt_ = self.output(r1)  # output 
        vt = vt_ + pin.to(vt_.device)  # velocity
        if self.pos_err:
            v1 = v1 + self.dt * (xref.to(vt_.device) - vt)  # integrated output error (position)
        else:
            v1 = xref.to(vt_.device) - vt  # direct output error (velocity)
        return x1, r1, vt_, v1, (in1, in2, in3_s)

    # GET VELOCITY OUTPUT (NOT ERROR)
    def get_output(self, testl1):
        return self.output(testl1)

    # RUN MODEL
    def forward(
        self,
        X,
        Xpert,
        Xref,
        local_learning_rule=None,
        fb_in=True,
        use_transpose=False,
        analysis=False,
        compute_grad=False,
        compute_hessian=False,
    ):
        gradients = []
        bptt = BPTT(lr=0.0)  # just for comparison of grads
        criterion = nn.MSELoss(reduction="none")  # hard coded for now
        # init_weights = self.rnn.weight_hh_l0.data.clone()
        self.batch_size = X.size(1)
        x1 = self.init_hidden()
        self.x0 = x1
        r1 = self.nonlin(x1)
        v1 = self.output(r1)
        # what to save
        hidden1 = [r1]
        poserr = [v1 * 0]
        raw_output = [v1]
        # initial variables needed for learning rules
        self.presum_alt = self.alpha * r1
        self.prepostsum = torch.zeros((x1.shape[1], x1.shape[1]), device=x1.device)
        # simulate time
        local_alignments = []
        local_alignments_2 = []
        hessian_eigvals = []
        local_grads = []
        r_inputs = []
        hidden_raw = []
        hidden_raw.append(x1)
        r_inputs.append(
            (torch.zeros_like(x1), r1 @ self.rnn.weight_hh_l0.T, torch.zeros_like(x1))
        )
        control = []
        for j in range(X.size(0)):
            if local_learning_rule.__class__.__name__ in ['BPTT_2']:
                print(j)
            # save previous time step for RFLO
            r1_prev = torch.clone(r1)
            if j < self.delay:
                x1, r1, vt, v1, r_input = self.f_step(
                    X[j], x1, r1, poserr[0] * 0, v1, Xpert[j], Xref[j], fb_in
                )
            else:
                error = poserr[j - self.delay]
                if self.error_type == "loss":
                    error = torch.as_tensor(
                        [criterion(error, error * 0).detach().mean()]
                    ).to(error.device)
                    if poserr[0].shape[1] == 2:
                        error = torch.stack(
                            [
                                error,
                                torch.tensor([0.0], device=error.device),
                            ]
                        ).T
                elif self.error_type == "abs_error":
                    error = abs(error)
                elif self.error_type == "d_loss":
                    error = torch.as_tensor(
                        [criterion(error, error * 0).detach().mean()]
                    ).to(error.device)
                    prev_error = poserr[j - self.delay - 1]  # HACK
                    error = error - torch.as_tensor(
                        [criterion(prev_error, prev_error * 0).detach().mean()]
                    ).to(
                        error.device
                    )  # difference between current and previous
                    if poserr[0].shape[1] == 2:
                        error = torch.stack(
                            [
                                error,
                                torch.tensor([0.0], device=error.device),
                            ]
                        ).T
                elif self.error_type == "error":
                    error = error

                x1, r1, vt, v1, r_input = self.f_step(
                    X[j], x1, r1, error, v1, Xpert[j], Xref[j], fb_in
                )
                
            # adaptation
            if local_learning_rule is not None:
                fb = v1[0] # poserr[j - self.delay][0]  # as batch == 1 # this was not correct - the error is in the past!
                # fb_nofb = v1_nofb[0]  # as batch == 1
                # update learning variables
                local_learning_rule.update(model=self, r1 = r1, r1_prev = r1_prev, x1 = x1)
                # get local gradient

                if local_learning_rule.__class__.__name__ in ['BPTT', 'BPTT_2', 'SAM']:
                    mse_loss = criterion(fb, fb * 0).mean()
                    inputs = (X[j], x1, r1, error, v1, Xpert[j], Xref[j], fb_in)
                    dw = local_learning_rule.get_recurrent_grads(
                        self, mse_loss, inputs
                    )
                else:
                    dw = local_learning_rule.get_recurrent_grads(
                        self, fb, use_transpose=use_transpose
                    )

                    if compute_grad:
                        # calculate local alignment
                        mse_loss = criterion(fb, fb * 0).mean()
                        true_grad = bptt.get_recurrent_grads(
                            self, mse_loss )
                        alignment_j = torch.nn.functional.cosine_similarity(
                            dw.detach().clone().flatten(),
                            true_grad.detach().clone().flatten(),
                            dim=0,
                        )
                        # print(j, alignment_j)
                        local_alignments.append(alignment_j)
                        if compute_hessian:
                            true_grad = -1 * true_grad
                            hessian = []
                            for g in true_grad.flatten():
                                gg, = torch.autograd.grad(g, self.rnn.weight_hh_l0, retain_graph=True)
                                hessian.append(gg.flatten().detach())
                            hessian = torch.stack(hessian) # |theta| x |theta|
    
                            he = torch.linalg.eigvals(hessian)
                            real = [h.real for h in he]
                            # assert the first value is the largest one
                            # assert real[0] == max(real)
                            # print(j, real[0], max(real))
                            hessian_eigvals.append([real[0], max(real)])

                            # compute approximate second order gradient
                            second_order_grad = (
                                - torch.pinverse(
                                    torch.add(hessian, torch.eye(hessian.shape[0],device=hessian.device), alpha=1e-3)
                                    ) @ true_grad.detach().flatten()
                                ).view_as(true_grad)


                            # calculate alignement between local and global
                            alignment_j2 = torch.nn.functional.cosine_similarity(
                                dw.detach().clone().flatten(),
                                second_order_grad.detach().clone().flatten(),
                                dim=0,
                            )
                            # print(j, alignment_j2)
                            local_alignments_2.append(alignment_j2)
                        
                # save local gradient
                local_grads.append(dw.detach().clone())
                # update weights
                self.rnn.weight_hh_l0.data = (
                    self.rnn.weight_hh_l0.data.detach() + local_learning_rule.lr * dw.detach()
                )
            # save for later
            hidden1.append(r1)
            hidden_raw.append(x1)
            raw_output.append(vt)
            poserr.append(v1)
            control.append(r_input[2])
            if j != 0 and analysis:
                feedback = r_input[2]
                gradients.append(
                    torch.stack(
                        [
                            torch.autograd.grad(
                                criterion(feedback, feedback * 0).mean(),
                                hidden1[-2],
                                retain_graph=True,
                            )[0],
                            torch.autograd.grad(
                                criterion(v1, v1 * 0).mean(),
                                hidden1[-1],
                                retain_graph=True,
                            )[0],
                            torch.autograd.grad(
                                criterion(v1, v1 * 0).mean(),
                                hidden_raw[-1],
                                retain_graph=True,
                            )[0],
                        ]
                    )
                )
            r_inputs.append(
                (r_input[0].detach(), r_input[1].detach(), r_input[2].detach())
            )

        hidden1 = torch.stack(hidden1) if not analysis else hidden1
        poserr = torch.stack(poserr)
        raw_output = torch.stack(raw_output)
        hidden_raw = (
            hidden_raw if analysis else torch.stack(hidden_raw).detach()
        )
        control = torch.stack(control)
        if analysis:
            gradients = torch.stack(gradients)

        extras = {}
        extras["local_grads"] = local_grads
        extras["local_alignments"] = local_alignments
        extras["local_alignments_2"] = local_alignments_2
        extras["hessian_eigvals"] = hessian_eigvals
        extras["raw_output"] = raw_output
        extras["r_inputs"] = r_inputs
        extras["gradients"] = gradients
        extras["hidden_raw"] = hidden_raw
        extras["control"] = control

        return poserr, hidden1, extras
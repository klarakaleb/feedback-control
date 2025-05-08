"""
Toolbox for model definition.
"""
from numpy import False_
import torch
import torch.nn as nn


class LearningRule:
    """Make learning rule selection more modular."""

    def __init__(self, lr):
        self.lr = lr

    def update(self, **kwargs):
        pass

    def get_recurrent_grads(self):
        pass


class BPTT(LearningRule):
    def __init__(self, lr):
        super().__init__(lr)

    def get_recurrent_grads(self, model, loss, inputs=None):
        # here we get the gradients
        dw, = torch.autograd.grad(
            loss,
            model.rnn.weight_hh_l0,
            retain_graph=True,
            create_graph=True,
        )
        return - dw


class FED(LearningRule):
    def __init__(self, lr):
        super().__init__(lr)

    def update(self, model, r1, r1_prev, x1):
        # here we update the eligibility traces
        # TODO: is this correct?
        model.presum_alt = model.alpha * r1 + (1 - model.alpha) * model.presum_alt

    def get_recurrent_grads(self, model, fb, use_transpose=False):
        # here we get the gradients
        if use_transpose:
            w = model.output.weight
        else:
            w = (model.feedback.weight).T
        dw = torch.outer(fb @ w, model.presum_alt[0])  
        return dw


class FED_t(FED):
    def __init__(self, lr):
        super().__init__(lr)

    def get_recurrent_grads(self, model, fb, use_transpose=True):
        # here we get the gradients
        dw = torch.outer(fb @ model.output.weight, model.presum_alt[0])
        return dw


class RFLO(LearningRule):
    def __init__(self, lr):
        super().__init__(lr)

    def update(self, model, r1, r1_prev, x1):
        # here we update the eligibility traces
        model.prepostsum = (
            model.alpha * torch.outer(model.nonlin_der(x1[0]), r1_prev[0])
            + (1 - model.alpha) * model.prepostsum
        )

    def get_recurrent_grads(self, model, fb, use_transpose=False):
        # here we get the gradients
        if use_transpose:
            w = model.output.weight
        else:
            w = (model.feedback.weight).T
        dw = ((fb @ w) * model.prepostsum.T).T
        return dw


class RFLO_t(RFLO):
    def __init__(self, lr):
        super().__init__(lr)

    def get_recurrent_grads(self, model, fb, use_transpose=True):
        # here we get the gradients
        dw = (
            torch.outer(
                fb @ model.output.weight,
                torch.ones(model.prepostsum[0].shape[0], device=fb.device),
            )
            * model.prepostsum[0]
        )
        return dw
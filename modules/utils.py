import torch
import numpy as np

def relu_der(x):
    return (x > 0) * 1


# from https://github.com/omarschall/vanilla-rtrl/blob/366a8a3d675db28eca090019dad9d78f5a825ab2/utils.py#L67
def normalized_dot_product(a, b):
    """Calculates the normalized dot product between two numpy arrays, after
    flattening them."""

    a_norm = torch.linalg.norm(a)
    b_norm = torch.linalg.norm(b)

    if a_norm > 0 and b_norm > 0:
        return torch.dot(a.flatten(), b.flatten()) / (a_norm * b_norm)
    else:
        return 0


# from https://github.dev/meulemansalex/deep_feedback_control/
def nullspace(A, tol=1e-12):
    U, S, V = torch.svd(A, some=False)
    if S.min() >= tol:
        null_start = len(S)
    else:
        null_start = int(len(S) - torch.sum(S < tol))

    V_null = V[:, null_start:]
    return V_null


# from https://github.dev/meulemansalex/deep_feedback_control/
def get_jacobian(input, output):
    if isinstance(input, torch.Tensor):
        input = [input]

    output_flat = output.view(-1)
    numel_input = 0
    for input_tensor in input:
        numel_input += input_tensor.numel()
    jacobian = torch.Tensor(output.numel(), numel_input)
    for i, output_elem in enumerate(output_flat):
        gradients = torch.autograd.grad(
            output_elem,
            input,
            retain_graph=True,
            create_graph=False,
            only_inputs=True,
        )
        jacobian_row = torch.cat([g.contiguous().view(-1).detach() for g in gradients])
        jacobian[i, :] = jacobian_row
    shape = list(output.shape)
    shape.append(-1)
    jacobian = jacobian.view(shape)
    return jacobian


def get_jacobian_2(input, output):

    jacobian = torch.autograd.grad(
        output,
        input,
        retain_graph=True,
        create_graph=False,
        only_inputs=True,
    )

    return jacobian


# from https://github.dev/meulemansalex/deep_feedback_control/
def get_nullspace_relative_norm(A, weights_update_flat):

    x = weights_update_flat

    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    A_null = nullspace(A)
    x_null_coordinates = A_null.cuda().t().mm(x.cuda())
    ratio = x_null_coordinates.norm() / x.norm()

    return ratio


def calc_dw(dinit):
    code_order = [
        "wihl0b",
        "whhl0b",
        "rnn0bto0",
        "wihl0",
        "whhl0",
        "wihl1",
        "whhl1",
        "wout",
        "bout",
    ]
    # code_names = ['->Upstream','Upstream rec.','Upstream->PMd','->PMd','PMd rec.','PMd->M1','M1','Output','Output bias']
    # code = {'wihl0':'rnn_l0.weight_ih_l0',
    #         'whhl0':'rnn_l0.weight_hh_l0',
    #         'wihl1':'rnn_l1.weight_ih_l0',
    #         'whhl1':'rnn_l1.weight_hh_l0',
    #         'wout':'output.weight','bout':'output.bias',
    #         'wihl0b':'rnn_l0b.weight_ih_l0',
    #         'whhl0b':'rnn_l0b.weight_hh_l0',
    #         'rnn0bto0':'rnn0bto0.weight'}
    dif = []
    dim = []
    for k in code_order:
        if k.split("_")[-1] == "mask":
            continue
        w = dinit["params1"][k]
        w2 = dinit["params2"][k]
        dif.append(np.median(abs((w2 - w) / (w) * 100)))
        try:
            _, e, _ = np.linalg.svd(w2 - w)
            pr = np.sum(e.real) ** 2 / np.sum(e.real ** 2)
            dim.append(pr)
        except:
            dim.append(np.nan)
    return dif, dim


def calc_dim(dw):
    try:
        _, e, _ = np.linalg.svd(dw)
        dim = np.sum(e.real) ** 2 / np.sum(e.real ** 2)
    except:
        dim = np.nan
    return dim
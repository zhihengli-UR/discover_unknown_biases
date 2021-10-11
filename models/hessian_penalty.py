import torch


def hessian_penalty(G, z, k=2, epsilon=0.1, reduction=torch.max, return_separately=False, G_z=None, interfaceGAN=False, **G_kwargs):
    """
    Official PyTorch Hessian Penalty implementation.
    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.
    :param G: Function that maps input z to either a tensor or a list of tensors (activations)
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>
    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        # G_z = G(z, **G_kwargs)
        if not interfaceGAN:
            G_z = G(latent=z)
        else:
            G_z = G(z, **G_kwargs)
    rademacher_size = torch.Size((k, *z.size()))  # (k, N, z.size())
    xs = epsilon * rademacher(rademacher_size, device=z.device)
    second_orders = []
    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(G, z, x, G_z, epsilon, interfaceGAN=interfaceGAN, **G_kwargs)
        second_orders.append(central_second_order)  # Appends a tensor with shape equal to G(z).size()
    loss = multi_stack_var_and_reduce(second_orders, reduction, return_separately)  # (k, G(z).size()) --> scalar
    return loss


def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device)
    x.random_(0, 2)  # Creates random tensor of 0s and 1s
    x[x == 0] = -1  # Turn the 0s into -1s
    return x


def multi_layer_second_directional_derivative(G, z, x, G_z, epsilon, interfaceGAN=False, **G_kwargs):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    # G_to_x = G(z + x, **G_kwargs)
    # G_from_x = G(z - x, **G_kwargs)


    if not interfaceGAN:
        G_to_x = G(latent=z + x)
        G_from_x = G(latent=z - x)
    else:
        G_to_x = G(z + x, **G_kwargs)
        G_from_x = G(z - x, **G_kwargs)

    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)  # (k, N, C, H, W)
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)  # (N, C, H, W)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty


def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]

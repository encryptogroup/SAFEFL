import math

import torch
import torch.nn.functional as F

import numpy as np
import random


def no_byz(v, net, lr, f, device):
    """
    No attack is performed therefore the gradients are simply returned.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    return v


def trim_attack(v, net, lr, f, device):
    """
    Local model poisoning attack against the trimmed mean aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    vi_shape = tuple(v[0].size())
    v_tran = torch.cat(v, dim=1)
    maximum_dim, _ = torch.max(v_tran, dim=1, keepdim=True)
    minimum_dim, _ = torch.min(v_tran, dim=1, keepdim=True)
    direction = torch.sign(torch.sum(torch.cat(v, dim=1), dim=-1, keepdim=True))
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim
    # let the malicious clients (first f clients) perform the attack
    for i in range(f):
        random_12 = (1. + torch.rand(*vi_shape)).to(device)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v


def krum_attack(v, net, lr, f, device):
    """
    Local model poisoning attack against the krum aggregation rule.
    Based on the description in https://arxiv.org/abs/1911.11815
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    threshold = 1e-5

    n = len(v)
    w_re = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0)
    d = v[0].size()[0]
    dist = torch.zeros((n, n)).to(device)
    for i in range(n):  # compute euclidean distance of benign to benign devices
        for j in range(i + 1, n):
            d = torch.norm(lr * v[i] - lr * v[j], p=2)
            dist[i, j], dist[j, i] = d, d

    dist_benign_sorted, _ = torch.sort(dist[f:, f:])
    min_dist = torch.min(torch.sum(dist_benign_sorted[:, 0:(n - f - 1)], dim=-1))

    dist_w_re = []
    for i in range(f, n):
        dist_w_re.append(torch.norm(lr * v[i], p=2))
    max_dist_w_re = torch.max(torch.stack(dist_w_re))

    max_lambda = min_dist / ((n - 2 * f - 1) * torch.sqrt(d)) + max_dist_w_re / torch.sqrt(d)

    actual_lambda = max_lambda
    sorted_dist, _ = torch.sort(dist, dim=-1)
    update_before = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
    while actual_lambda > threshold:
        for i in range(f):
            v[i] = - actual_lambda * torch.sign(update_before)

        dist = torch.zeros((n, n)).to(device)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.norm(v[i] - v[j])
                dist[i, j], dist[j, i] = d, d
        sorted_dist, _ = torch.sort(dist, dim=-1)
        global_update = v[torch.argmin(torch.sum(sorted_dist[:, 0:(n - f - 1)], dim=-1))]
        if torch.equal(global_update, v[0]):
            break
        else:
            actual_lambda = actual_lambda / 2

    return v


def fltrust_attack(v, net, lr, f, device):
    """
    Local model poisoning attack against the fltrust aggregation rule.
    Based on the specification in https://arxiv.org/abs/2012.13995 originally named adaptive attack.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    norm_p = 2.0  # Euclidean norm
    n = len(v) - 1
    V = 10
    Q = 10
    std = 0.5  # sigma^2
    gamma = 0.005
    eta = 0.01

    eps = 1e-9

    e = [v[i] / torch.norm(v[i], p=norm_p) for i in range(n)]
    e_0 = v[-1] / torch.norm(v[-1], p=norm_p)
    c = [torch.dot(torch.squeeze(e_i), torch.squeeze(e_0)) for e_i in e]

    sum_c_i = torch.sum(F.relu(torch.stack(c))) + eps
    temp = [F.relu(c[i]) / sum_c_i * e[i] for i in range(n)]
    weighted_sum_e = torch.sum(torch.cat(temp, dim=1), dim=1, keepdim=True)
    norm_g_0 = torch.norm(v[-1], p=norm_p)
    g = norm_g_0 * weighted_sum_e

    s = torch.sign(g)

    v_prime = trim_attack(v.copy(), net, lr, f, device)[0:f]
    e_prime = [v_prime_i / torch.norm(v_prime_i, p=norm_p) for v_prime_i in v_prime]

    def h(e_prime_prime):
        c_prime = [torch.dot(torch.squeeze(e_prime_i), torch.squeeze(e_0)) for e_prime_i in e_prime_prime]
        sum_c_prime = torch.sum(F.relu(torch.stack(c_prime))) + torch.sum(F.relu(torch.stack(c[f:]))) + eps

        model_difference = (weighted_sum_e
                            - torch.sum(
                    torch.cat([F.relu(c_prime[j]) * e_prime_prime[j] / sum_c_prime for j in range(0, f)], dim=-1),
                    dim=1, keepdim=True)
                            - torch.sum(torch.cat([F.relu(c[j]) * e[j] / sum_c_prime for j in range(f, n)], dim=-1),
                                        dim=1, keepdim=True))

        return norm_g_0 * torch.dot(torch.squeeze(s), torch.squeeze(model_difference))

    for _ in range(V):
        for i in range(f):
            for t in range(Q):
                u = torch.normal(mean=0, std=std, size=tuple(v[0].size())).to(device)
                grad_h = (h([e_prime[j] + gamma * u if j == i else e_prime[j] for j in range(f)]) - h(
                    e_prime)) / gamma * u
                e_prime[i] = e_prime[i] + eta * grad_h
                e_prime[i] = e_prime[i] / torch.norm(e_prime[i], p=norm_p)

    for i in range(f):
        v[i] = e_prime[i] * norm_g_0

    return v


def min_max_attack(v, net, lr, f, device):
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
    but refactored for clarity.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    deviation = grad_mean / torch.norm(grad_mean, p=2)  # decided to use unit_vec distance which was their default
    # but they also had the option to use sign and standard deviation
    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)  # determine max distance left side of optimization
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)

    for i in range(f):
        v[i] = mal_update[:, None]

    return v


def min_sum_attack(v, net, lr, f, device):
    """
    Local model poisoning attack from https://par.nsf.gov/servlets/purl/10286354
    The implementation is based of their repository (https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning)
    but refactored for clarity.
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    catv = torch.cat(v, dim=1)
    grad_mean = torch.mean(catv, dim=1)
    deviation = grad_mean / torch.norm(grad_mean, p=2)  # decided to use unit_vec distance which was their default
    # but they also had the option to use sign and standard deviation
    gamma = torch.Tensor([50.0]).float().to(device)
    threshold_diff = 1e-5
    gamma_fail = gamma
    gamma_succ = 0

    distances = []
    for update in v:
        distance = torch.norm(catv - update, dim=1, p=2) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    # finding optimal gamma according to algorithm 1
    while torch.abs(gamma_succ - gamma) > threshold_diff:
        mal_update = (grad_mean - gamma * deviation)
        distance = torch.norm(catv - mal_update[:, None], dim=1, p=2) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            gamma_succ = gamma
            gamma = gamma + gamma_fail / 2
        else:
            gamma = gamma - gamma_fail / 2

        gamma_fail = gamma_fail / 2
    mal_update = (grad_mean - gamma_succ * deviation)

    for i in range(f):
        v[i] = mal_update[:, None]

    return v


def label_flipping_attack(each_worker_label, f, num_labels):
    """
    Data poisoning attack which changes the labels of the training data on the malicious clients.
    each_worker_label: data labels of workers
    f: number of malicious clients, where the first f are malicious
    num_labels: highest label number
    """
    for i in range(f):
        each_worker_label[i] = num_labels - each_worker_label[i] - 1

    return each_worker_label


def scaling_attack_insert_backdoor(each_worker_data, each_worker_label, dataset, f, device):
    """
    Data poisoning attack which inserts backdoor patterns into the training data on the malicious clients.
    The attack is based on the description in https://arxiv.org/abs/2012.13995
    The trigger pattern is from https://arxiv.org/abs/1708.06733
    each_worker_data: data of each worker
    each_worker_label: labels of the data of each worker
    dataset: name of the dataset used in training
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    if dataset == "HAR":
        attacker_chosen_target_label = 1
        for i in range(f):
            p = 1 - np.random.rand(1)[0]  # sample random number from (0,1]
            number_of_backdoored_images = math.ceil(p * each_worker_data[i].size(dim=0))
            benign_images = each_worker_data[i].size(dim=0)

            # expand list of images with number of backdoored images and copy all benign images
            expanded_data = torch.zeros(benign_images + number_of_backdoored_images,
                                        each_worker_data[i].size(dim=1)).to(device)

            for n in range(benign_images):
                expanded_data[n] = each_worker_data[i][n]

            # duplicate images and add pattern trigger
            for j in range(number_of_backdoored_images):
                # Currently first image is selected every time
                random_number = random.randrange(0, each_worker_data[i].size(dim=0))
                backdoor = each_worker_data[i][random_number, :]
                for k in range(len(backdoor)):
                    if (k + 1) % 20 == 0:
                        backdoor[k] = 0
                expanded_data[benign_images + j] = backdoor

            # replace data of compromised worker with expanded data
            each_worker_data[i] = expanded_data

            # expand list of labels with number of backdoored images with attacker chosen target label
            each_worker_label[i] = torch.tensor(each_worker_label[i].tolist() +
                                   [attacker_chosen_target_label for i in range(number_of_backdoored_images)]).to(device)
    else:
        raise NotImplementedError

    return each_worker_data, each_worker_label


def scaling_attack_scale(v, net, lr, f, device):
    """
    Second part of the scaling attack which scales the gradients of the malicious clients to increase their impact.
    The attack is based on the description in https://arxiv.org/abs/2012.13995
    v: list of gradients
    net: model
    lr: learning rate
    f: number of malicious clients, where the first f are malicious
    device: device used in training and inference
    """
    scaling_factor = len(v)
    for i in range(f):
        v[i] = v[i] * scaling_factor
    return v


def add_backdoor(data, labels, dataset):
    """
    Adds backdoor to a provided list of data examples.
    The trigger pattern is from https://arxiv.org/abs/1708.06733
    data: list data examples
    labels: list of the labels of data
    dataset: name of the dataset from which data was sampled
    """
    if dataset == "HAR":
        attacker_chosen_target_label = 1

        # add pattern trigger
        for i in range(data.size(dim=0)):
            for k in range(data.size(dim=1)):
                if (k + 1) % 20 == 0:
                    data[i][k] = 0

            # expand list of labels with number of backdoored images with attacker chosen target label
            for i in range(len(labels)):
                labels[i] = attacker_chosen_target_label
    else:
        raise NotImplementedError

    return data, labels

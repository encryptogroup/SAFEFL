from __future__ import print_function

import aggregation_rules
import numpy as np
import random
import argparse
import attacks
import data_loaders

import os
import math
import subprocess

import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt


def parse_args():
    """
    Parses all commandline arguments.
    """
    parser = argparse.ArgumentParser(description="SAFEFL: MPC-friendly framework for Private and Robust Federated Learning")

    ### Model and Dataset
    parser.add_argument("--net", help="net", type=str, default="lr")
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="HAR")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--p", help="bias probability of class 1 in server dataset", type=float, default=0.1)

    ### Training
    parser.add_argument("--niter", help="# iterations", type=int, default=2000)
    parser.add_argument("--nworkers", help="# workers", type=int, default=30)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.25)
    parser.add_argument("--gpu", help="no gpu = -1, gpu training otherwise", type=int, default=-1)
    parser.add_argument("--seed", help="seed", type=int, default=1)
    parser.add_argument("--nruns", help="number of runs for averaging accuracy", type=int, default=1)
    parser.add_argument("--test_every", help="testing interval", type=int, default=50)

    ### Aggregations
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fedavg")

    # FLOD
    parser.add_argument("--flod_threshold", help="hamming distance threshold as fraction of total model parameters", type=float, default=0.5)

    # FLAME
    parser.add_argument("--flame_epsilon", help="epsilon for differential privacy in FLAME", type=int, default=3000)
    parser.add_argument("--flame_delta", help="delta for differential privacy in FLAME", type=float, default=0.001)

    # DNC
    parser.add_argument("--dnc_niters", help="number of iterations to compute good sets in DnC", type=int, default=5)
    parser.add_argument("--dnc_c", help="filtering fraction, percentage of number of malicious clients filtered", type=float, default=1)
    parser.add_argument("--dnc_b", help="dimension of subsamples must be smaller, then the dimension of the gradients", type=int, default=2000)

    ### Attacks
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=6)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no", choices=["no", "trim_attack", "krum_attack",
                            "scaling_attack", "fltrust_attack", "label_flipping_attack", "min_max_attack", "min_sum_attack"])

    ### MP-SPDZ
    parser.add_argument('--mpspdz', default=False, action='store_true', help='Run example in multiprocess mode')
    parser.add_argument("--port", help="port for the mpc servers", type=int, default=14000)
    parser.add_argument("--chunk_size", help="data amount send between client and server at once", type=int, default=200)
    parser.add_argument("--protocol", help="protocol used in MP-SPDZ", type=str, default="semi2k",
                        choices=["semi2k", "spdz2k", "replicated2k", "psReplicated2k"])
    parser.add_argument("--players", help="number of computation parties", type=int, default=2)
    parser.add_argument("--threads", help="number of threads per computation party in MP-SPDZ", type=int, default=1)
    parser.add_argument("--parallels", help="number of parallel computation for each thread", type=int, default=1)
    parser.add_argument('--always_compile', default=False, action='store_true', help='compiles program even if it was already compiled')

    return parser.parse_args()


def get_device(device):
    """
    Selects the device to run the training process on.
    device: -1 to only use cpu, otherwise cuda if available
    """
    if device == -1:
        ctx = torch.device('cpu')
    else:
        ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(ctx)
    return ctx


def get_net(net_type, num_inputs, num_outputs=10):
    """
    Selects the model architecture.
    net_type: name of the model architecture
    num_inputs: number of inputs of model
    num_outputs: number of outputs/classes
    """
    if net_type == "lr":
        import models.lr as lr
        net = lr.LinearRegression(input_dim=num_inputs, output_dim=num_outputs)
        print(net)
    else:
        raise NotImplementedError
    return net


def get_byz(byz_type):
    """
    Gets the attack type.
    byz_type: name of the attack
    """
    if byz_type == "no":
        return attacks.no_byz
    elif byz_type == 'trim_attack':
        return attacks.trim_attack
    elif byz_type == "krum_attack":
        return attacks.krum_attack
    elif byz_type == "scaling_attack":
        return attacks.scaling_attack_scale
    elif byz_type == "fltrust_attack":
        return attacks.fltrust_attack
    elif byz_type == "label_flipping_attack":
        return attacks.no_byz
    elif byz_type == "min_max_attack":
        return attacks.min_max_attack
    elif byz_type == "min_sum_attack":
        return attacks.min_sum_attack
    else:
        raise NotImplementedError


def get_protocol(protocol, players):
    """
    Returns the shell script name and number of players for the protocol.
    protocol: name of the protocol
    players: number of parties
    """
    if players < 2:
        raise Exception("Number of players must at least be 2")

    if protocol == "semi2k":
        return "semi2k.sh", players

    elif protocol == 'spdz2k':
        return "spdz2k.sh", players

    elif protocol == "replicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for replicated2k")
        return "ring.sh", 3

    elif protocol == "psReplicated2k":
        if players != 3:
            raise Exception("Number of players must be 3 for psReplicated2k")
        return "ps-rep-ring.sh", 3

    else:
        raise NotImplementedError


def evaluate_accuracy(data_iterator, net, device, trigger, dataset):
    """
    Evaluate the accuracy and backdoor success rate of the model. Fails if model output is NaN.
    data_iterator: test data iterator
    net: model
    device: device used in training and inference
    trigger: boolean if backdoor success rate should be evaluated
    dataset: name of the dataset used in the backdoor attack
    """
    correct = 0
    total = 0
    successful = 0

    net.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_iterator):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)

            if not torch.isnan(outputs).any():
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.shape[0]
            else:
                print("NaN in output of net")
                raise ArithmeticError

            if trigger:     # backdoor attack
                backdoored_inputs, backdoored_targets = attacks.add_backdoor(inputs, targets, dataset)
                backdoored_outputs = net(backdoored_inputs)
                if not torch.isnan(backdoored_outputs).any():
                    _, backdoored_predicted = backdoored_outputs.max(1)
                    successful += backdoored_predicted.eq(backdoored_targets).sum().item()
                else:
                    print("NaN in output of net")
                    raise ArithmeticError

    success_rate = successful / total
    acc = correct / total
    if trigger:
        return acc, success_rate
    else:
        return acc, None


def plot_results(runs_test_accuracy, runs_backdoor_success, test_iterations, niter):
    """
    Plots the evaluation results.
    runs_test_accuracy: accuracy of the model in each iteration specified in test_iterations of every run
    runs_backdoor_success: backdoor success of the model in each iteration specified in test_iterations of every run
    test_iterations: list of iterations the model was evaluated in
    niter: number of iteration the model was trained for
    """
    test_acc_std = []
    test_acc_list = []
    backdoor_success_std = []
    backdoor_success_list = []

    # insert (0,0) as starting point for plot and calculate mean and standard deviation if multiple runs were performed
    if args.nruns == 1:
        if args.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=0)
            backdoor_success_list = runs_backdoor_success
            backdoor_success_std = [0 for i in range(0, len(runs_backdoor_success))]
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=0)
        test_acc_list = runs_test_accuracy
        test_acc_std = [0 for i in range(0, len(runs_test_accuracy))]
    else:
        if args.byz_type == "scaling_attack":
            runs_backdoor_success = np.insert(runs_backdoor_success, 0, 0, axis=1)
            backdoor_success_list = np.mean(runs_backdoor_success, axis=0)
            backdoor_success_std = np.std(runs_backdoor_success, axis=0)
        runs_test_accuracy = np.insert(runs_test_accuracy, 0, 0, axis=1)
        test_acc_std = np.std(runs_test_accuracy, axis=0)
        test_acc_list = np.mean(runs_test_accuracy, axis=0)

    test_iterations.insert(0, 0)
    # Print accuracy and backdoor success rate in array form to console
    print("Test accuracy of runs:")
    print(repr(runs_test_accuracy))
    if args.byz_type == "scaling_attack":
        print("Backdoor attack success rate of runs:")
        print(repr(runs_backdoor_success))

    # Determine in which iteration in what run the highest accuracy was achieved.
    # Also print overall mean accuracy and backdoor success rate
    max_index = np.unravel_index(runs_test_accuracy.argmax(), runs_test_accuracy.shape)
    if args.nruns == 1:
        print("Run 1 in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] * 50, runs_test_accuracy.max()))
    else:
        print("Run %02d in iteration %02d had the highest accuracy of %0.4f" % (max_index[0] + 1, max_index[1] * 50, runs_test_accuracy.max()))
        print("The average final accuracy was: %0.4f with an overall average:" % (test_acc_list[-1]))
        print(repr(test_acc_list))
        if args.byz_type == "scaling_attack":
            print("The average final backdoor success rate was: %0.4f with an overall average:" % backdoor_success_list[-1])
            print(repr(backdoor_success_list))
    # Generate plot with two axis displaying accuracy and backdoor success rate over the iterations
    if args.byz_type == "scaling_attack":
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        accuracy_plot = ax1.plot(test_iterations, test_acc_list, color='C0', label='accuracy')
        ax1.fill_between(test_iterations, test_acc_list - test_acc_std, test_acc_list + test_acc_std, color='C0')
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Backdoor success rate')
        backdoor_plot = ax2.plot(test_iterations, backdoor_success_list, color='C1', label='Backdoor success rate')
        ax2.fill_between(test_iterations, backdoor_success_list - backdoor_success_std, backdoor_success_list + backdoor_success_std, color='C1')
        ax2.set_ylim(0, 1)

        lns = accuracy_plot + backdoor_plot
        labels = [l.get_label() for l in lns]
        plt.legend(lns, labels, loc=0)
        plt.xlim(0, niter)
        plt.title("Test Accuracy + Backdoor success: " + args.net + ", " + args.dataset + ", " + args.aggregation + ", " + args.byz_type + ", nruns " + str(args.nruns))
        plt.grid()
        plt.show()
    # Generate plot with only the accuracy as one axis over the iterations
    else:
        plt.plot(test_iterations, test_acc_list, color='C0')
        plt.fill_between(test_iterations, test_acc_list - test_acc_std, test_acc_list + test_acc_std, color='C0')
        plt.title("Test Accuracy: " + args.net + ", " + args.dataset + ", " + args.aggregation + ", " + args.byz_type + ", nruns " + str(args.nruns))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.xlim(0, niter)
        plt.ylim(0, 1)
        plt.grid()
        plt.show()


def weight_init(m):
    """
    Initializes the weights of the layer with random values.
    m: the layer which gets initialized
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=2.24)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def main(args):
    """
    The main function that runs the entire training process of the model.
    args: arguments defining hyperparameters
    """

    # setup
    device = get_device(args.gpu)
    num_inputs, num_outputs, num_labels = data_loaders.get_shapes(args.dataset)
    byz = get_byz(args.byz_type)

    # Print all arguments
    paraString = ('dataset: p' + str(args.p) + '_' + str(args.dataset) + ", server_pc: " + str(args.server_pc) + ", bias: " + str(args.bias)
                  + ", nworkers: " + str(args.nworkers) + ", net: " + str(args.net) + ", niter: " + str(args.niter) + ", lr: " + str(args.lr)
                  + ", batch_size: " + str(args.batch_size) + ", nbyz: " + str(args.nbyz) + ", attack: " + str(args.byz_type)
                  + ", aggregation: " + str(args.aggregation) + ", FLOD_threshold: " + str(args.flod_threshold)
                  + ", Flame_epsilon: " + str(args.flame_epsilon) + ", Flame_delta: " + str(args.flame_delta) + ", Number_runs: " + str(args.nruns)
                  + ", DnC_niters: " + str(args.dnc_niters) + ", DnC_c: " + str(args.dnc_c) + ", DnC_b: " + str(args.dnc_b)
                  + ", MP-SPDZ: " + str(args.mpspdz) + ", Port: "+ str(args.port) + ", Chunk_size: " + str(args.chunk_size)
                  + ", Protocol: " + args.protocol + ", Threads: " + str(args.threads) + ", Parallels: " + str(args.parallels)
                  + ", Seed: " + str(args.seed) + ", Test Every: " + str(args.test_every))
    print(paraString)

    # saving iterations for averaging
    runs_test_accuracy = []
    runs_backdoor_success = []
    test_iterations = []
    backdoor_success_list = []

    # model
    net = get_net(args.net, num_outputs=num_outputs, num_inputs=num_inputs)
    net = net.to(device)
    num_params = torch.cat([xx.reshape((-1, 1)) for xx in net.parameters()], dim=0).size()[0]  # used for FLOD to determine threshold
    # loss
    softmax_cross_entropy = nn.CrossEntropyLoss()

    # perform parameter checks
    if args.dnc_b > num_params and args.aggregation == "divide_and_conquer":
        args.dnc_b = num_params  # check for condition in description and fix possible error
        print("b was larger than the dimension of gradients. Set to dimension of gradients for correctness!")

    if args.dnc_c * args.nbyz >= args.nworkers and args.aggregation == "divide_and_conquer":
        print("DnC removes all gradients during his computation. Lower c or nbyz, or increase number of devices.")

    if args.server_pc == 0 and (args.aggregation in ["fltrust", "flod", "flare"] or args.byz_type == "fltrust_attack"):
        raise ValueError("Server dataset size cannot be 0 when aggregation is FLTrust, MPC FLTrust, FLOD or attack is fltrust attack")

    if args.dataset == "HAR" and args.nworkers != 30:
        raise ValueError("HAR only works for 30 workers!")

    # compile server programm for aggregation in MPC
    if args.mpspdz:
        script, players = get_protocol(args.protocol, args.players)
        args.script, args.players = script, players

        if args.aggregation == "fedavg":
            args.filename_server = "mpc_fedavg_server"
            num_gradients = args.nworkers
        elif args.aggregation == "fltrust":
            args.filename_server = "mpc_fltrust_server"
            num_gradients = args.nworkers + 1
        else:
            raise NotImplementedError

        os.chdir("mpspdz")

        args.full_filename = f'{args.filename_server}-{args.port}-{num_params}-{num_gradients}-{args.niter}-{args.chunk_size}-{args.threads}-{args.parallels}'

        if not os.path.exists('./Programs/Bytecode'):
            os.mkdir('./Programs/Bytecode')
        already_compiled = len(list(filter(lambda f : f.find(args.full_filename) != -1, os.listdir('./Programs/Bytecode')))) != 0

        if args.always_compile or not already_compiled:
            # compile mpc program, arguments -R 64 -X were chosen so that every protocol works
            os.system('./compile.py -R 64 -X ' + args.filename_server + ' ' + str(args.port) + ' ' + str(num_params) + ' ' + str(num_gradients) + ' ' + str(args.niter) + ' ' + str(args.chunk_size) + ' ' + str(args.threads) + ' ' + str(args.parallels))

        # setup ssl keys
        os.system('Scripts/setup-ssl.sh ' + str(args.players))
        os.system('Scripts/setup-clients.sh 1')

        os.chdir("..")

    # perform multiple runs
    for run in range(1, args.nruns+1):
        grad_list = []
        test_acc_list = []
        test_iterations = []
        backdoor_success_list = []
        server_process = None

        # fix the seeds for deterministic results
        if args.seed > 0:
            args.seed = args.seed + run - 1
            torch.cuda.manual_seed_all(args.seed)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            np.random.seed(args.seed)

        net.apply(weight_init)  # initialization of model

        # set aggregation specific variables
        if args.aggregation == "shieldfl":
            previous_global_gradient = 0  # important for ShieldFL, all other aggregation rules don't need it
            previous_gradients = []
        elif args.aggregation == "foolsgold":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in range(args.nworkers)]   # client gradient history for FoolsGold
        elif args.aggregation == "contra":
            gradient_history = [torch.zeros(size=(num_params, 1)).to(device) for i in range(args.nworkers)]   # client gradient history for CONTRA
            reputation = torch.ones(size=(args.nworkers, )).to(device)                                        # reputation scores for CONTRA
            cos_dist = torch.zeros((args.nworkers, args.nworkers), dtype=torch.double).to(device)               # pairwise cosine similarity for CONTRA
        elif args.aggregation == "romoa":
            # don't know why they initialize it like this
            previous_global_gradient = torch.cat([param.clone().detach().flatten() for param in net.parameters()]).reshape(-1, 1) + torch.normal(mean=0, std=1e-7, size=(num_params, 1)).to(device)
            sanitization_factor = torch.full(size=(args.nworkers, num_params), fill_value=(1 / args.nworkers)).to(device)  # sanitization factors for Romoa

        train_data, test_data = data_loaders.load_data(args.dataset, args.seed)  # load the data

        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = data_loaders.assign_data(train_data, args.bias, device,
            num_labels=num_labels, num_workers=args.nworkers, server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=args.seed)

        # perform data poisoning attacks
        if args.byz_type == "label_flipping_attack":
            each_worker_label = attacks.label_flipping_attack(each_worker_label, args.nbyz, num_labels)
        elif args.byz_type == "scaling_attack":
            each_worker_data, each_worker_label = attacks.scaling_attack_insert_backdoor(each_worker_data, each_worker_label, args.dataset, args.nbyz, device)

        print("Data done")

        # start FLTrust computation parties
        if args.mpspdz:
            os.chdir("mpspdz")

            print("Starting Computation Parties")
            # start computation servers using a child process to run in parallel
            server_process = subprocess.Popen(["./run_aggregation.sh", args.script, args.full_filename, str(args.players)])

            os.chdir("..")


        with torch.no_grad():
            # training
            for e in range(args.niter):
                net.train()

                # perform local training for each worker
                for i in range(args.nworkers):
                    minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=args.batch_size, replace=False)

                    net.zero_grad()
                    with torch.enable_grad():
                        output = net(each_worker_data[i][minibatch])
                        loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                        loss.backward()

                    grad_list.append([param.grad.clone().detach() for param in net.parameters()])

                # compute server update and append it to the end of the list
                if args.aggregation in ["fltrust", "flod"] or args.byz_type == "fltrust_attack":
                    net.zero_grad()
                    with torch.enable_grad():
                        output = net(server_data)
                        loss = softmax_cross_entropy(output, server_label)
                        loss.backward()
                    grad_list.append([torch.clone(param.grad) for param in net.parameters()])

                # perform the aggregation
                if args.mpspdz:
                    aggregation_rules.mpspdz_aggregation(grad_list, net, args.lr, args.nbyz, byz, device, param_num=num_params, port=args.port, chunk_size=args.chunk_size, parties=args.players)

                elif args.aggregation == "fltrust":
                    aggregation_rules.fltrust(grad_list, net, args.lr, args.nbyz, byz, device)

                elif args.aggregation == "fedavg":
                    data_sizes = [x.size(dim=0) for x in each_worker_data]
                    aggregation_rules.fedavg(grad_list, net, args.lr, args.nbyz, byz, device, data_sizes)

                elif args.aggregation == "krum":
                    aggregation_rules.krum(grad_list, net, args.lr, args.nbyz, byz, device)

                elif args.aggregation == "trim_mean":
                    aggregation_rules.trim_mean(grad_list, net, args.lr, args.nbyz, byz, device)

                elif args.aggregation == "median":
                    aggregation_rules.median(grad_list, net, args.lr, args.nbyz, byz, device)

                elif args.aggregation == "flame":
                    aggregation_rules.flame(grad_list, net, args.lr, args.nbyz, byz, device, epsilon=args.flame_epsilon, delta=args.flame_delta)

                elif args.aggregation == "shieldfl":
                    previous_global_gradient, previous_gradients = aggregation_rules.shieldfl(grad_list, net, args.lr, args.nbyz, byz, device, previous_global_gradient, e, previous_gradients)

                elif args.aggregation == "flod":
                    aggregation_rules.flod(grad_list, net, args.lr, args.nbyz, byz, device, threshold=math.floor(num_params * args.flod_threshold))

                elif args.aggregation == "divide_and_conquer":
                    aggregation_rules.divide_and_conquer(grad_list, net, args.lr, args.nbyz, byz, device, niters=args.dnc_niters, c=args.dnc_c, b=args.dnc_b)

                elif args.aggregation == "foolsgold":
                    gradient_history = aggregation_rules.foolsgold(grad_list, net, args.lr, args.nbyz, byz, device, gradient_history=gradient_history)

                elif args.aggregation == "contra":
                    gradient_history, reputation, cos_dist = aggregation_rules.contra(grad_list, net, args.lr, args.nbyz, byz, device, gradient_history=gradient_history, reputation=reputation, cos_dist=cos_dist, C=1)

                elif args.aggregation == "signguard":
                    aggregation_rules.signguard(grad_list, net, args.lr, args.nbyz, byz, device, seed=args.seed)

                elif args.aggregation == "flare":
                    aggregation_rules.flare(grad_list, net, args.lr, args.nbyz, byz, device, server_data)

                elif args.aggregation == "romoa":
                    sanitization_factor, previous_global_gradient = aggregation_rules.romoa(grad_list, net, args.lr, args.nbyz, byz, device, F=sanitization_factor, prev_global_update=previous_global_gradient, seed=args.seed)

                else:
                    raise NotImplementedError

                del grad_list
                grad_list = []
                # evaluate the model accuracy
                if (e + 1) % args.test_every == 0:
                    test_accuracy, test_success_rate = evaluate_accuracy(test_data, net, device, args.byz_type == "scaling_attack", args.dataset)
                    test_acc_list.append(test_accuracy)
                    test_iterations.append(e)
                    if args.byz_type == "scaling_attack":
                        backdoor_success_list.append(test_success_rate)
                        print("Iteration %02d. Test_acc %0.4f. Backdoor success rate: %0.4f" % (e, test_accuracy, test_success_rate))
                    else:
                        print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))

        if args.mpspdz:
            server_process.wait()   # wait for process to exit

        # Append accuracy and backdoor success rate to overall runs list
        if len(runs_test_accuracy) > 0:
            runs_test_accuracy = np.vstack([runs_test_accuracy, test_acc_list])
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = np.vstack([runs_backdoor_success, backdoor_success_list])
        else:
            runs_test_accuracy = test_acc_list
            if args.byz_type == "scaling_attack":
                runs_backdoor_success = backdoor_success_list
        if args.byz_type == "scaling_attack":
            print("Run %02d/%02d done with final accuracy: %0.4f and backdoor success rate: %0.4f" % (run, args.nruns, test_acc_list[-1], backdoor_success_list[-1]))
        else:
            print("Run %02d/%02d done with final accuracy: %0.4f" % (run, args.nruns, test_acc_list[-1]))


    del test_acc_list
    test_acc_list = []


if __name__ == "__main__":
    args = parse_args()     # parse arguments
    main(args)      # call main with parsed arguments

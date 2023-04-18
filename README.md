# SAFEFL: MPC-friendly framework for Private and Robust Federated Learning
This project implements several federated learning aggregation rules and attacks.
We added support for linear regression on the HAR dataset.

Additionally, we implemented [FLTrust](https://arxiv.org/abs/2012.13995) and [FedAvg](https://arxiv.org/abs/1602.05629) in the [MP-SPDZ](https://eprint.iacr.org/2020/521) Multi-Party Computation Framework.

The project is based on code by the authors of [FLTrust](https://arxiv.org/abs/2012.13995) and follows their general structure. 
The original code is available [here](https://people.duke.edu/~zg70/code/fltrust.zip) and uses the machine learning framework MXNet.
We adapted the existing code to use PyTorch and extended it.

## Aggregation rules
The following aggregation rules have been implemented:

- [FedAvg](https://arxiv.org/abs/1602.05629)
- [Krum](https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- [Trimmed mean](https://arxiv.org/abs/1803.01498)
- [Median](https://arxiv.org/abs/1803.01498)
- [FLTrust](https://arxiv.org/abs/2012.13995)
- [FLAME](https://arxiv.org/abs/2101.02281)
- [FLOD](https://eprint.iacr.org/2021/993)
- [ShieldFL](https://ieeexplore.ieee.org/document/9762272)
- [DnC](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/)
- [FoolsGold](https://arxiv.org/abs/1808.04866)
- [CONTRA](https://par.nsf.gov/servlets/purl/10294585)
- [FLARE](https://dl.acm.org/doi/10.1145/3488932.3517395)
- [Romoa](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_23)
- [SignGuard](https://arxiv.org/abs/2109.05872)


All aggregation rules are located in _aggregation_rules.py_ as individual functions and operate on the local gradients and not on the actual local models.
Working with the gradients or working with the models is equivalent as long as the global model is known. 
All aggregation rules that normally work on the local models have been modified to work on the local gradients instead.

To add an aggregation rule you can add the implementation in _aggregation_rules.py_.
To actually use the aggregation rule during training you must also add a case for the aggregation rule in the main function of the _main.py_ file.
This calls the aggregation rule and must return the aggregated gradients.

## Attacks
To evaluate the robustness of the aggregation rules we also added the following attacks.

- [Label Flipping](https://proceedings.mlr.press/v20/biggio11.html)
- [Krum Attack](https://arxiv.org/abs/1911.11815)
- [Trim Attack](https://arxiv.org/abs/1911.11815)
- [Scaling Attack](https://arxiv.org/abs/2012.13995)
- [FLTrust Attack](https://arxiv.org/abs/2012.13995)
- [Min-Max Attack](https://par.nsf.gov/servlets/purl/10286354)
- [Min-Sum Attack](https://par.nsf.gov/servlets/purl/10286354)

The implementation of the attacks are all located in _attacks.py_ as individual functions.

To add a new attack the implementation can simply be added as a new function in this file. For attacks that are called during the aggregation the signature of the function 
must be the same format as the other attacks. This is because the attack function call in the training process is overloaded 
and which attack is called is only determined during runtime. 
The attack name must also be added to the get_byz function in _main.py_.
Attacks that only manipulate training data just need to be called before the training starts and don't need a specific signature.

## Models
We implemented multiclass linear regression classifier.

The model is in a separate file in the _models_ folder of this project. 

To add models a new file containing a class that defines this classifier must be added.
Additionally, in _main.py_ the _get_net_ function needs to be expanded to enable the selection of this model.

## Datasets
We implemented the [HAR](https://upcommons.upc.edu/handle/2117/20897) dataset and as it is not implemented by PyTorch per default. 
It must be downloaded with the provided loading script in the _data_ folder.

Adding a new dataset requires adding the loading to the _load_data_ function in _data_loading.py_. 
This can either be simply done by adding an existing dataloader from PyTorch or requires custom data loading like in the case with the HAR dataset.
Additionally, the size of the data examples and the number of classes need to be added to the _get_shapes_ function to properly configure the model.
Furthermore, the _assign_data_ function needs to be extended to enable assigning the test and train data to the individual clients.
Should the evaluation require running the new dataset with the scaling attack, which adds backdoor trigger patterns to the data examples the following functions also need to be extended:

- _scaling_attack_insert_backdoor_
- _add_backdoor_

Both of these are located in _attacks.py_.

## Multi-Party Computation
To run the MPC Implementation the [code](https://github.com/data61/MP-SPDZ) for [MP-SPDZ](https://eprint.iacr.org/2020/521) needs to be downloaded separately using the installation script _mpc_install.sh_.
The following protocols are supported:
- Semi2k uses 2 or more parties in a semi-honest, dishonest majority setting
- [SPDZ2k](https://eprint.iacr.org/2018/482) uses 2 or more parties in a malicious, dishonest majority setting
- [Replicated2k](https://eprint.iacr.org/2016/768.pdf) uses 3 parties in a semi-honest, honest majority setting
- [PsReplicated2k](https://eprint.iacr.org/2019/164.pdf) uses 3 parties in a malicious, honest majority setting

# How to run?

The project can be simply cloned from git and then requires downloading the [HAR](https://upcommons.upc.edu/handle/2117/20897) dataset as described in the dataset section.

The project takes multiple command line arguments to determine the training parameters, attack, aggregation, etc. is used.
If no arguments are provided the project will run with the default arguments.
A description of all arguments can be displayed by executing:
```shell
python main.py -h
```
# Requirements
The project requires the following packages to be installed:

- Python 3.8.13 
- Pytorch 1.11.0
- Torchvision 0.12.0
- Numpy 1.21.5
- MatPlotLib 3.5.1
- HDBSCAN 0.8.28
- Perl 5.26.2

All requirements can be found in the _requirements.txt_.

# Credits
This project is based on code by Cao et al. the authors of [FLTrust](https://arxiv.org/abs/2012.13995) and is available [here](https://people.duke.edu/~zg70/code/fltrust.zip)

We thank the authors of [Romoa](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_23) for providing an implementation of their aggregation.

We used the [open-sourced](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning) implementations of the [Min-Max](https://par.nsf.gov/servlets/purl/10286354) and [Min-Sum](https://par.nsf.gov/servlets/purl/10286354) attack.

For the implementation of Flame we used the scikit-learn implementation of [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) by McInnes et al.

The MPC Framework MP-SPDZ was created by [Marcel Keller](https://github.com/data61/MP-SPDZ).

# License
[MIT](https://choosealicense.com/licenses/mit/)

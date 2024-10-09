# Exogenous Matching: Learning Good Proposals for Tractable Counterfactual Estimation

This repository contains the complete code for "[Exogenous Matching: Learning Good Proposals for Tractable Counterfactual Estimation](https://arxiv.org/abs/...)".

## Requirements

- Install Manually

    Ensure the installation of Conda, create a new Conda environment, and activate it:

    ```shell
    conda create -n exom python=3.11 -y
    conda activate exom
    ```

    Next, install the necessary dependencies for this repository:

    ```shell
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
    conda install tensorboard lightning -c conda-forge -y
    pip install seaborn tueplots zuko==1.1.0
    ```

    All learning components are constructed based on the Torch framework, and we recommend utilizing GPU accelerators to expedite sampling. The implementation of normalizing flows and masking mechanisms is from Zuko, with this work involving minor modifications to their source code.

- Install from `environment.yml`

    ```shell
    conda env create -f environment.yml
    ```

## Training & Evaluation

To ensure reproducibility, we have archived the randomly generated ground truth and test cases from the experiments, which need to be extracted in advance:

```shell
bash make_pt.sh
```

All experimental results will be stored in `/output` directory.

- Complete Trials

    The experiments presented in the paper can be conducted using the following script, which performs both training and testing simultaneously:

    ```shell
    # Convergence
    bash script/counterfactual/loglikelihood_effective/train.sh

    # Comparison
    bash script/counterfactual/sample_validation/train_rs.sh
    bash script/counterfactual/sample_validation/train_ce.sh
    bash script/counterfactual/sample_validation/train_nis.sh
    bash script/counterfactual/sample_validation/train_exom.sh

    # Combinations of SCMs and Density Estimation Models
    bash script/counterfactual/sample_validation/train_exom_full.sh
    bash script/counterfactual/sample_validation/train_exom_full2.sh

    # Ablation
    bash script/counterfactual/mask_validation/train.sh
    bash script/counterfactual/reduce_validation/train.sh
    ```

- Single Trial

    If a complete experiment is time-consuming, a simple test can be conducted by specifying parameters directly via the command line, for instance:

    ```shell
    bash script/counterfactual/utils/sample/train_exom.sh\
        -n "{TRIAL_NAME}"\
        -s "{SCM_NAME}"\
        -j "{JOINT_NUM}"\
        -m "{DENSITY_MODEL}"\
        -c "{ENCODE_PATTERN}"\
        -k "{MASK_PATTERN}"\
        -r "{REDUCE_PATTERN}"\
        -t "{TRANSFORMS / COMPONENTS}"\
        -d "{HIDDEN_LAYERS}"\
        -sv "{EVAL_DATA_SIZE}" -bv "{EVAL_BATCH_SIZE}"\
        -e "{EPOCH}" -sd "{SEED}"
    ```

    The specific meanings and rules of the parameters are as follows (the randomly specified counterfactual process here is $\mathfrak{Q}_*^\mathcal{B}$):

    | Flag | Parameter            | Description                                                                                                                                                                                                                                                                                                                                                                                                                             | Example        |
    |------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
    | -n   | TRIAL_NAME           | A string representing the experiment name.                                                                                                                                                                                                                                                                                                                                                                                            | `"MY_TRIAL"`   |
    | -s   | SCM_NAME             | The name of the ground truth SCM for the experiment. Options include: `"chain_lin_3", "chain_nlin_3", "chain_lin_4", "chain_lin_5", "collider_lin", "fork_lin", "fork_nlin", "largebd_nlin", "simpson_nlin", "simpson_symprod", "triangle_lin", "triangle_nlin", "back_door", "front_door", "m", "napkin", "fairness", "fairness_xw", "fairness_xy", "fairness_yw"` | `"simpson_nlin"` |
    | -j   | JOINT_NUM            | The size of $\|s\|$ in the random counterfactual process $\mathfrak{Q}_*^\mathcal{B}$. Options include: `1, 3, 5`.                                                                                                                                                                                                                                                                                                             | `5`            |
    | -m   | DENSITY_MODEL        | Density model of the conditional distribution. Options include: `"gmm", "maf", "nsf", "ncsf", "nice", "naf", "sospf"`.                                                                                                                                                                                                                                                                                                               | `"maf"`        |
    | -c   | ENCODE_PATTERN       | Defines the encoding method. Single encodings include: `"e", "t", "e+t", "w_e", "w_t", "w_e+w_t"`, where `"e"` represents Evidence (observed terms in counterfactual variables), `"t"` represents Treatment (intervention terms in counterfactual variables), the prefix `"w_"` indicates the corresponding indicator, and `"+"` means summation. Multiple encodings are separated by spaces, corresponding to the concatenation of vectors. | `"e+t w_e w_t"` |
    | -k   | MASK_PATTERN         | Defines the masking method. The final vector to be learned as a condition is “encoding $\otimes$ mask”. Single masks include: `"fc", "em", "mb", "mb1", "mb2"`. Where `"fc"` is no mask, `"em"` masks only the observed or intervened encodings, and `"mb", "mb1", "mb2"` are three different masking methods for (counterfactual) Markov boundaries in Ablation. Multiple encodings are separated by spaces, corresponding to the concatenation of vectors, and must correspond one-to-one with each single encoding in ENCODE_PATTERN.         | `"mb1 mb1 em"`  |
    | -r   | REDUCE_PATTERN       | Defines the aggregation method. Optional aggregation methods include: `"attn", "wsum", "sum", "concat"`.                                                                                                                                                                                                                                                                                                                            | `"attn"`       |
    | -t   | TRANSFORMS / COMPONENTS | Number of transforms in the flow-based model; or number of components in GMM.                                                                                                                                                                                                                                                                                                                                                             | `5`            |
    | -d   | HIDDEN_LAYERS        | Hidden layers. Options include: `"64", "128", "64x2", "96x2", "128x2", "192x2", "256x2", "32x3", "64x3", "96x3"`.                                                                                                                                                                                                                                                                                                                    | `"64x2"`       |
    | -sv  | EVAL_DATA_SIZE       | Size of the test and validation set.                                                                                                                                                                                                                                                                                                                                                                                                   | `1024`         |
    | -bv  | EVAL_BATCH_SIZE      | Batch size for the test and validation set.                                                                                                                                                                                                                                                                                                                                                                                          | `32`           |
    | -e   | EPOCH                | Maximum training epoch.                                                                                                                                                                                                                                                                                                                                                                                                                 | `200`          |
    | -sd  | SEED                 | Seed for everything.                                                                                                                                                                                                                                                                                                                                                                                                                    | `0`            |

    An example of parameter assignment is:
    
    ```shell
    bash script/counterfactual/utils/sample/train_exom.sh\
        -n "MY_TRIAL"\
        -s "simpson_nlin"\
        -j 5\
        -m "maf"\
        -c "e+t w_e w_t"\
        -k "mb1 mb1 em"\
        -r "attn"\
        -t 5\
        -d "64x2"\
        -sv 1024 -bv 32\
        -e 200 -sd 0
    ```

- Trials with Proxy SCMs
    
    Regarding the experiments on the proxy SCM, it is necessary to first train the proxy SCM, and then run the corresponding scripts.

    ```shell
    # Training CausalNFs
    bash script/proxy_scm/causal_nf/make/train.sh

    # Training NCMs
    bash script/proxy_scm/ncm/make/train.sh

    # Counterfactual Estimation on CausalNFs
    bash script/counterfactual/density_estimation/train_rs.sh
    bash script/counterfactual/density_estimation/train_exom.sh
    bash script/counterfactual/density_estimation/train_rs_from_causal_nf.sh
    bash script/counterfactual/density_estimation/train_exom_from_causal_nf.sh

    # Counterfactual Estimation on NCMs
    bash script/counterfactual/density_estimation/train_rs.sh
    bash script/counterfactual/density_estimation/train_exom.sh
    bash script/counterfactual/density_estimation/train_rs_from_causal_nf.sh
    bash script/counterfactual/density_estimation/train_exom_from_causal_nf.sh
    ```

## Results

The `.py` scripts in the subdirectory `script/figure/drawers` are used to extract results from `/output` and generate tables and figures. The extracted experimental results are consolidated into `.csv` files located in the subdirectory under `script/figure/drawers`, while the tables and figures generated by the scripts are saved in `script/figure/tabs` and `script/figure/imgs`, respectively.

## File Structure

### Common Components

The source code in this directory is widely used throughout the project, providing fundamental components.

```shell
common
├── graph
│   ├── causal.py
│   └── utils.py
└── scm
    ├── eq.py
    ├── scm.py
    └── utils.py
```

The `graph` directory pertains to topics related to causal graphs. Within this directory, `causal.py` provides a basic implementation of ADMG (Acyclic Directed Mixed Graphs) and Augmented Graphs, while `utils.py` offers several utility functions.

The `scm` directory deals with Structural Causal Models (SCM). Specifically, `eq.py` allows for the encapsulation of causal equations using a decorator pattern. The `scm.py` file defines and implements two types of stochastic SCM objects: a generalized SCM and a TensorSCM designed for all experiments. Additionally, `utils.py` within this directory provides various utility functions.

### Datasets

All datasets are synthetic, and we represent the sampling and stochastic counterfactual processes described in the paper as datasets.

```shell
dataset/
├── evidence
│   ├── fixed_samplers
│   ├── mcar_samplers
│   ├── query_samplers
│   ├── batched_evidence.py
│   ├── batched_evidence_custom.py
│   ├── batched_evidence_joint.py
│   ├── batched_evidence_sampler.py
│   ├── dataset.py
│   ├── evidence.py
│   ├── evidence_custom.py
│   ├── evidence_joint.py
│   ├── evidence_sampler.py
│   ├── markov_boundary.py
│   └── type_map.py
├── synthetic
│   ├── markovian_diffeomorphic
│   ├── recursive_continuous
│   ├── regional_canonical
│   └── dataset.py
└── utils.py
```

The code in the `evidence` directory describes stochastic counterfactual processes and their sampling results as datasets. There are two sampling methods: one involves lazy sampling by each worker when synthesizing a batch is needed, and the other involves uniform pre-sampling followed by workers reading the samples (these contents are prefixed with "batched_", but the final results are identical to the former). `evidence_sampler.py` serves as a comprehensive integration of all samplers, while `dataset.py` provides dataset encapsulation. `evidence.py`, `evidence_joint.py`, and `evidence_custom.py` offer structured representations of the sampling results. The implementation of stochastic counterfactual processes can be found in the directories `fixed_samplers` (only for testing), `mcar_samplers` (corresponding to $\mathfrak{Q}_*^\mathcal{B}$), and `query_samplers` (corresponding to $\mathfrak{Q}_*^\mathcal{Q}$).

The code within the `synthetic` directory encompasses the specific parameter forms of all fully-specified SCMs, with the exception of `regional_canonical`, which necessitates the use of random generation (with generated parameters are saved in `.pt` files in a subdirectory).

### Models

All code related to model construction is included in the `model` directory.

```shell
model/
├── counterfactual
│   ├── cross_entropy
│   ├── exo_match
│   ├── gaussian_sample
│   ├── naive_sample
│   ├── neural
│   ├── query
│   ├── jctf_estimator.py
│   ├── metric.py
│   └── utils.py
├── proxy_scm
│   ├── causal_nf
│   └── ncm
└── zuko
    ├── autoregressive.py
    ├── continuous.py
    ├── coupling.py
    ├── mixture.py
    ├── neural.py
    ├── nn.py
    ├── polynomial.py
    └── spline.py
```

The `zuko` directory contains our modified versions of the GMMs and normalizing flows within the Zuko library. These modifications enable the models to support batch-wise conditional masking.

The `proxy_scm` directory contains our replication code for CausalNF and NCM, respectively.

The `counterfactual` directory contains the primary experiments and comparative objects of the article, specifically the counterfactual estimations. The `jctf_estimator.py` encapsulates the tools and methods for counterfactual estimation under the Lightning Module framework which includes the validation process, ultimately producing estimates of counterfactual probabilities. Within the subdirectories, `naive_sample` corresponds to Rejection Sampling (RS), `cross_entropy` corresponds to Cross-entropy based Importance Sampling (CEIS), `neural` corresponds to Neural Importance Sampling (NIS), and `exo_match` corresponds to Exogenous Matching (EXOM). The `query` directory implements the estimation of specific $\mathcal{L}_3$ expressions in counterfactual estimation, relying on the counterfactual probability estimates provided by `jctf_estimator.py`.

### Scripts

All scripts for configurations, and code related to generating figures and tables are included in the `script` directory.

```shell
script/
├── counterfactual
│   ├── density_estimation
│   ├── effect_estimation
│   ├── loglikelihood_effective
│   ├── mask_validation
│   ├── reduce_validation
│   ├── sample_validation
│   ├── utils
│   ├── config_dataset.py
│   ├── config_evidence.py
│   ├── config_model.py
│   ├── config_sampler.py
│   ├── config_scm.py
│   └── train.py
├── figure
│   ├── drawers
│   ├── imgs
│   └── tabs
├── proxy_scm
│   ├── causal_nf
│   └── gan_ncm
├── config.py
├── config_dataloader.py
└── config_trainer.py
```

The directories `counterfactual` and `proxy_scm` contain all the scripts used for experiments. The `figure` directory contains preprocessed output data (as .csv files), scripts for generating figures and figures, and the resulting files.

## Citation

To be continue....
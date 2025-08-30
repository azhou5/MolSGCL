# MolSGCLr
Datasets for molecular property prediction are small relative to the vast chemical space, making generalization from limited experiments a central challenge. We present Mol-SGCL, Molecular Substructure-Guided Contrastive Learning, a method that shapes the latent space of molecular property prediction models to align with science-based priors. We hypothesize that engineering inductive biases directly into the representation space encourages models to learn chemical principles rather than overfitting to spurious correlations. Concretely, Mol-SGCL employs a triplet loss that pulls a molecule’s representation toward representations of plausibly causal substructures and pushes it away from implausibly causal ones. Plausibility is either defined by domain-specific rules in Mol-SGCL_Rules or by a large language model in Mol-SGCL_LLM. To stress-test out-of-distribution (OOD) generalization under data scarcity, we construct modified Therapeutics Data Commons tasks that minimize train–test similarity and cap the training set at 150 molecules. On these OOD splits, both Mol-SGCL_rules and Mol-SGCL_LLM consistently outperform baselines, indicating that \name promotes invariant feature learning and enhances model generalizability in data-limited regimes. We further demonstrate that Mol-SGCL transfers successfully to Minimol, a state-of-the-art molecular property prediction model, highlighting that the approach is not tied to a specific architecture. We envision that Mol-SGCL could be extended beyond molecular property prediction to any setting where inputs can be decomposed into substructures whose presence, absence, or configuration has a causal influence on the target label. 
The environments used can be found at:
chemprop_environment.yml (for chemprop models) and minimol_environment.yml for minimol models. 

![Overview](./overview.png)


# Repo Structure

*chemprop_environment.yml* and *minimol_environment.yml* can be used to recreate the conda environments used in this study.
The custom data-splits in this study are in *data*
*chemprop_custom* is a clone of the chemprop package, with substantial changes in the models/model.py (where the triplet loss is defined), as well as the *data/datasets.py, and data/datapoints.py file. 
In *code*:
- *dmpnn_mol_sgcl.py* is the file to run the DMPNN version of Mol-SGCL 
- *run dmpnn_mol_sgcl_llm.sh* is a bash script that can be used to run dmpnn_mol_sgcl.py for Mol-SGCL_Rules. It is pre-written to run the lipophilicity evaluation. 
- *run dmpnn_mol_sgcl_llm.sh* is a bash script that can be used to run dmpnn_mol_sgcl.py for Mol-SGCL_Rules. It is pre-written to run the lipophilicity evaluation. 
- *minimol_triplet_model.py* is where Minimol-Mol-SGCL model is defined 
- *minimol_triplet_runner.py* is the python file that sets up the evaluation of the Minimol-Mol-SGCL file
- *run_minimol_mgscl* is a bash script that can be used to run dmpnn_mol_sgcl.py for Mol-SGCL_Rules. It is pre-written to run the lipophilicity evaluation. 
- *plausibility_utils.py* contains the manually defined rules as python functions 
- *model_utils.py* contains several assorted utility functions
- *get_rationale_for_plausibility.py* and *get_rationale_regression.py* run a MCTS to return substructures.
- *LLM_assisted_plausibility* is a folder that contains:
    - *get_plausibility.py*: This uses Deepseek to produce plausibility labels. 
    - *molecular_description_image.py*: The uses Deepseek to generate a natural language description of molecules. 

# To run Mol-SGCL_Rules: 

1. Run the run_dmpnn_mol_sgcl_rules.sh file. Make sure to activate your chemprop environment. 

# To run Mol-SGCL_LLM: 

1. Create a file in your Add your DeepSeek API key to to a .env file in the project root. 

2. Run the run_dmpnn_mol_sgcl_rules.sh file. Make sure to activate your chemprop environment. 

# To run Mol-SGCL_Minimol:

1. Run the run_minimol_msgcl.sh file. Make sure to activate your minimol environment. 

2. This analysis is preloaded with the lipophilicity task. 






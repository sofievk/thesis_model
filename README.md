# Thesis Project



This repository contains the code, notebooks, and supporting data necessary for the replication of my thesis project, where I adapt an existing IAM of the energy transition, the GHKT model, by changing the production function and adding mineral constraints to the production of low-carbon energy. 



It is a work in progress. The current folder structure is as follows:



Thesis Project/

│

├── notebooks/   # Jupyter notebooks

│    └── ghkt_replication\_replication.ipynb     # Notebook to run replication in Python/Jupyter
 
│    └── thesis_full\thesis_full.ipynb           # Notebook to run replication in Python/Jupyter

├── src/                          # All MATLAB code and model files

│   ├── ghkt\_original/           # Original GHKT model files

│   ├── ghkt\_replication/        # My replication of the GHKT model 

│   ├── ghkt\_mineral\_only/      # GHKT with added mineral constraints

│   ├── newpf\_full/              # New model with new production function and mineral constraints

│   └── newpf\_nomin/             # New production function only

│
│

├── .gitignore                 # Specifies files/folders not tracked by Git

├── environment.yml            # Conda environment specification for Python/Jupyter

└── README.md                  # Project description and instructions


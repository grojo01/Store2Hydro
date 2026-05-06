# Store2Hydro in PyPSA-EUR
This repository hosts the code for a ?Case-Study? performed in PyPSA-EUR (https://github.com/pypsa/pypsa-eur). The Case-Study is part of the Europe Horizon Project "Store2Hydro" (https://www.store2hydro.com/), which researches the possibility of retrofitting hydro plants to Pumped Hydro Storage systems to improve systems with high renewable generation.
As part of work package 4 this work focuses on the specific needs of energy systems incorperating retrofitted hydropower plants.

## Method
For this project the PyPSA-EUR source code was expanded by a endogeneous retrofitting function for hydropower plants. This was done by expanding @add_electricity.py, adding a retrofitted PHS plant with the same data as the hydro plant with zero power and by expanding @solve_network.py with the  binary constraint z. The nominal power of the hydro is set by the optimizer according to
- P_nom_hydro = (1 - z) * p_nom_orig_hydro
- P_nom_retrofit = z * p_nom_orig_hydro

Parameters for the retrofit scenario can be set in the confi.yaml file under renewable.hydro:
    retrofit_to_phs:
      enable: true
      capital_cost: 100  
      max_share: 0.5

## Project structure
The method is implemented in pypsa-eur. Project-related data is organized in a superior folder structere:
- /code: has all the analyzing tools for networks and data
- /data: has additional data used in the beginning of the research (preloadede networks) -> not in use
- /pypsa-eur: has all the source code
    - also has snakemake logs: detailing runtimes and mistakes
- /results: has all the results from analysis
- /submission_scripts: has submission-script templates and submission scripts for HPC via SLURM

## Getting started
It is recommended to use high performance clusters (HPC) to run PyPSA-EUR. Smaller models can be executed on personal computers!

For a detailed installation guide of pypsa-eur check out their documentation (https://pypsa-eur.readthedocs.io/en/latest/installation.html). The following instructions is for once the project is set up on the HPC

1. Adjust config.yaml 
2. Adjust config.scenario.yaml (if different scenarios are to be simulated)

The following may differ from your workflow depending on what system you. This project´s computing was performed on Slurm, which is most commonly in use for hpc.

3. adjust jobscript file (see @job-template.sh.template -> for jobs .sh-files are used)
4. submit jobscript via "sbatch submission_scripts/job-name.sh" in the terminal of the HPC

## Acknowledgements
Of course we thank the Team of **Prof. Tom Brown** and all the Contributors of **PyPSA-EUR** for creating this impressive dataset.
Furthermore we want to thank **DFG** and **University of Bayreuth** for providing the HPC infrastructure as calculations were performed using the **festus-cluster** of the Bayreuth Centre for High Performance Computing (https://www.bzhpc.uni-bayreuth.de), funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 523317330.
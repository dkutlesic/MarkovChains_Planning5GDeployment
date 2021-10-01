# Deploying a 5G Network in a country
This repository provides the solution of team _"X/|Eb"_ to the project for graduate course _"Markov Chains and Algorithmic Applications"_. (COM-516) at EPFL.

The goal of the project is to plan the deployment of 5G network in a country with given cities (coordinates and populations are known). For this problem, we designed several solutions to how the problem can be seen as a Markov chain. Next, we developed Metropolis-Hastings algorithm to optimize the objective function. Finally, we did a comparative analysis among different proposed chains and suggested a network deployment plan.


A complete description of the project is given in _project_description.pdf_.

Report of our work is given in _paper.pdf_.

The notebook _main.ipynb_ contains all visualizations presented in our report. The source code for Metropolis-Hastings algorithm is in _./src_ as well as auxiliary functions. In our solution, besides the main task we carry out comparative analysis of four base chains. For this purpose, we prepared results of experiments for all chains with different $\lambda$, $\beta$ and its increase factor. This data is contained in _./chains_results_ as PyTorch tensors.

Authors:
Dubravka Kutlesic (@dkutlesic, dubravka.kutlesic@epfl.ch)
Aleksandr Timofeev (@TimofeevAlex, aleksandr.timofeev@epfl.ch)
Andrei Afonin (@AfoninAndrey, andrei.afonin@epfl.ch)

# Multiple Sequence Alignment using Deep Reinforcement Learning
### Bachelor-Thesis of Roman Joeres
#### at chair for Modelling and Simulation at Saarland University, Germany

## Structure of this Repository
### agent_training
Files to perform the training of the agents, providing trainer classes for both major parts of reinforcement learning 
agents. Those agents approximating a state or action value function using tables or function approximation (Table- and 
ValueAgent, summarized in FunctionTrainer) and those approximating a policy using neural nets (Policy- and 
ActorCriticAgents, summarized in PolicyTrainer).
### agents
Agents-Classes to represent the agents that are trained by the above described agents. Here is also the parent class of 
all agents (solver.py) that contains the basic functionality needed to perform alignments ans its extension to agents
using networks to solve this task. One agent here, not contained in the trainer-folder is the mcts agent, because in 
MCTS these is no training that is performed in the way it is performed on all the other agents. 
### data
Sequence files in FASTA-format to evaluate the agents on. The Folder consists mainly of three parts: \
First, the sequence-files using in the reference paper, signed as bX_XXX.fasta. Those consist all of DNA-Sequence data 
with different lengths of the sequences (up to approximately 1000 bases) and up to 12 sequences. \
Because these benchmark files are not very representative, I added two more files with 6 and 8 sequences of moderate 
length to get a better graduated evaluation of the performance of the agents on an intermediate number of sequences. 
Since these files are intended to be used for optimization they are labeled as oX_XXX.fasta. \
The third part comes from an missing type of sequences in the previous files. None of the (D)RL-using paper compared
their work to protein sequence alignments of the state-of-the-art tools like CLUSTAL, MAFFT and MUSCLE. The files 
containing protein sequences to align are labeled as pX_XXX.fasta.
### networks
In this folder the networks are defined that are used in the training of all function approximating agents that uses for 
their approximation of whatever function neural nets. There are three main parts: \
First, the FFNN-nets are, as they are named, just Vanilla Feed-Forward Networks that either have one output node if they 
are used as state-value function or they have as many output nodes as there are action to choose from. \
In the reinforce-networks we have for each network again one or two sub-networks, named policy or value net according to 
their function in the REINFORCE algorithm, either as baseline-state-value function or as policy net to predict actions.\
The actorcritic_networks are quite similar to the REINFORCE networks in that they also consists of two output-nets, 
but, different to the REINFORCE nets these nets have the body in common. That means, that the first one or two layer of
neurons are for both nets the same, depending on the size of the complete network.
### tests
Here some tests are provided to mainly test the functionality of the helper-classes in the utils folder.
### utils
This folder is more or less the heart of the repository as here the main environment is stored (wrapper.py) and the 
algorithm of how to aligning sequences and to store them is implemented (alignment.py and profile.py). But also helper 
classes of dynamic programming as the hash-align-table and the hash-q-table are stored. The first to save intermediate 
alignments that does not have to be recomputed to save computational time and the latter one is used as a Q-table int 
the tabular agents. Also an implementation of a prioritized-replay-buffer can be found. But also the utils used in many
sub-tasks of this theses are collected in utils.py and many hyperparameter and other constants are defined in 
constants.py. 
## run and execute:
The optimizer.py can be used to test different configurations of the hyperparameter per agent to find the optimal one. 
This can be applied agent-wise for the same sequence file or in a method that mixes the optimization- and run-scripts to
execute different configurations of different agents on different problem instances (sequence files) to find compare 
them. \
To run multiple configurations of multiple agents on multiple benchmarks the commandline interface can be used. It takes
arguments as intended in the help (-h) or take alternatively a json file, containing multiple configurations, ... .\
For more detailed description of the json-structure look at the json_tutorial.txt.
To start this program one has to call the msadrl.py script.
## Benchmarks and Comparison
The algorithms has been tested agaist some state-of-the-art algorithms, namly CLUSTAL, MAFFT and MUSCLE. The results of
the runs can be found as raw-data in the compasiton.csv file that has been produced by the benchmark_comparator.py 
script.\
It exists also an internal comparison of the agents against each other in order to find the best one and to find the 
best alignments on each used development benchmark. The results are stored json-encoded in the best_benches.json file 
and can also be analysed via the msadrl commandline interface as described in the help that will be returned after 
giving the -h or --help argument to the script.

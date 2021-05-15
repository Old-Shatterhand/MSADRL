# JSON-Configuration-Structure

This is an overview to the json-files one can provide to the commandline interface. This tutorial does not describe the 
structure of an json-file and focus mainly on the options to give to the agents and will present the json-internal 
structure used for this.

## Structure

The two main parts of this are lists of agent-objects and benchmark-objects. So, in the agents-objects the type and all
needed parameters for learning are set. Below, you can find a tabular overview of which attributes has to be set in 
order to use the full power of this option of determining training. 

| Name         | Table | Value | Policy  | ActorCritic | MCTS |
|--------------|-------|-------|---------|-------------|------|
| Games        | N     | N     | N       | N           | --   |
| Alpha        | [0,1] | [0,1] | [0,1]x2 | [0,1]x2     | --   |
| Gamma        | [0,1] | [0,1] | [0,1]x2 | [0,1]x2     | --   |
| Lambda       | [0,1] | [0,1] | --      | --          | --   |
| N            | [-2,) | [-2,) | --      | --          | --   |
| Baseline     | --    | --    | B       | --          | --   |
| Simulations  | --    | --    | --      | --          | N    |
| Rollouts     | --    | --    | --      | --          | N    |
| C            | --    | --    | --      | --          | R    |
| Look         | B     | B     | B       | B           | --   |
| Support      | B     | B     | B       | B           | --   |
| Optimize     | SP/C  | SP/C  | SP/C    | SP/C        | SP/C |
| Refinement   | B     | B     | B       | B           | B    |
| Progress     | B     | B     | B       | B           | B    |
| Graph        | B     | B     | B       | B           | --   |
| Notify       | B     | B     | B       | B           | B    |
| Individual   | B     | B     | B       | B           | B    |

"N" in a column means, that one has to provide a natural number. Except for parameter N. Natural numbers are also the 
only accepted parametrization if it is "[-2,)". That means any integer is accepted from -2 upwards including -2. \
"[0,1]" means, this parameter has to be a real number in the interval from 0 to 1. x2 stands for the two values to 
provide for a parameter to set this as different value and policy parameter \ 
"B" parameters has to be given as bool values (e.g. true, false). "R" parameters should be given as real number.\
Fields with "--" have no influence on the agent. \
The only mandatory field of each agent is the "Name" that determines the algorithm to use. \
These fields have to be defined according to the specification of json and as in the example below. \
\
The second part is about the benchmarks, that can be defined either as internal Benchmark-Annotation and will then get
full support in the prints by additional information on the behaviour of other classical reference algorithms. 
As, an alternative one can provide files with other instances of the multiple sequence alignments problem. In this case 
the other benchmarking algorithms are not executed and one can only compare the defined agents.

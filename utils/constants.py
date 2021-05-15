# information on the used benchmarks to be displayed after an exhaustive run of multiple agents in the runner/optimizer
seq_files = ['data/b1_hepatitis_c_virus.fasta',  # 0 DNA 10 x 212
             'data/b2_papio_anubis.fasta',  # 1 DNA 5 x 1093
             'data/b3_oxbench_469.fasta',  # 2 DNA 3 x 332
             'data/b4_oxbench_429.fasta',  # 3 DNA 12 x 171
             'data/b5_dataset_lgm.fasta',  # 4 DNA 3 x 39
             'data/b6_dataset_rlo.fasta',  # 5 DNA 3 x 93
             'data/b7_dataset_1.fasta',  # 6 DNA 3 x 129
             'data/o1_oxbench_414.fasta',  # 7 DNA 6 x 154
             'data/o2_oxbench_415.fasta',  # 8 DNA 4 x 126
             'data/p1_oxbench_433.fasta',  # 9 Pep 3 x 186
             'data/p2_oxbench_641t2.fasta',  # 10 Pep 3 x 271
             'data/p3_oxbench_34.fasta',  # 11 Pep 6 x 251
             'data/p4_oxbench_620.fasta']  # 12 Pep 15 x 330
types = ['DNA'] * 9 + ['Pep'] * 4
sizes = [['10', '212'], ['5', '1093'], ['3', '332'], ['12', '171'], ['3', '39'], ['3', '93'], ['3', '129'],
         ['6', '154'], ['4', '126'], ['3', '186'], ['3', '271'], ['6', '215'], ['15', '330']]
names = ['Hepatitis-C-Virus', 'Papio Anubis', 'Oxbench 469', 'Oxbench 429', 'LGM-Dataset', 'RLO-Dataset',
         'Dataset 1', 'Oxbench 414', 'Oxbench 415', 'Oxbench 433', 'Oxbench 641t2', 'Oxbench 34', 'Oxbench 620']
benchmarks = [[18627, 0.93, 18627, 0.93, 18627, 0.93, 18627, 0.93, 18627, 0.93],
              [18719, 0.82, 18860, 0.82, 18827, 0.82, 18860, 0.82, 1878, 0.83],
              [565, 0.45, 565, 0.45, 464, 0.37, 549, 0.35, 555, 0.36],
              [8668, 0.2, 10218, 0.2, 9575, 0.18, 10218, 0.19, 10965, 0.18],
              [345, 0.66, 345, 0.66, 345, 0.62, 345, 0.62, 348, 0.64],
              [486, 0.65, 486, 0.65, 480, 0.63, 471, 0.62, 479, 0.64],
              [167, 0.73, 0, 0, 149, 0.66, 134, 0.6, 137, 0.62],
              [0, 0, 0, 0, 1800, 0.31, 1806, 0.32, 1855, 0.33],
              [0, 0, 0, 0, 513, 0.31, 513, 0.31, 534, 0.33],
              [0, 0, 0, 0, 268, 0.24, 290, 0.29, 24, 0.28],
              [0, 0, 0, 0, 659, 0.38, 723, 0.42, 442, 0.38],
              [0, 0, 0, 0, 355, 0.02, 377, 0.03, -206, 0.03],
              [0, 0, 0, 0, 13858, 0.10, 13893, 0.10, 4207, 0.1]]

# different comparison algorithms
RL, DRL, CLUSTALW, MAFFT, MUSCLE = 0, 1, 2, 3, 4
REFERENCES = [RL, DRL, CLUSTALW, MAFFT, MUSCLE]

# specifying types of agents to use in testing them on instances of the multiple sequence alignment problem
TABLE_AGENT, VALUE_AGENT, POLICY_AGENT, ACTOR_CRITIC_AGENT, MCTS_AGENT, ALPHA_ZERO_AGENT = 5, 6, 7, 8, 9, 10
AGENTS = [TABLE_AGENT, VALUE_AGENT, POLICY_AGENT, ACTOR_CRITIC_AGENT, MCTS_AGENT]  # , ALPHA_ZERO_AGENT]

# assign indices to each sequence data file
B1, B2, B3, B4, B5, B6, B7, O1, O2, P1, P2, P3, P4 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

agent_names = {'Q': TABLE_AGENT, 'V': VALUE_AGENT, 'P': POLICY_AGENT, 'A': ACTOR_CRITIC_AGENT, 'M': MCTS_AGENT,
               '0': ALPHA_ZERO_AGENT}
benchs = {'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B6': B6, 'B7': B7,
          'O1': O1, 'O2': O2, 'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4}

# scores for alignment process
score_dna_match = 2
score_dna_mismatch = -1
score_dna_gap = -2
score_protein_gap = -5

# constants for alignments
DIAG, LEFT, UP = 1, 2, 4
GAP = '-'

# definition for MC-Learning and Learning using the lambda return
MC_LEARN = -1
LAMBDA_LEARN = -2

# scoring of the alignments
SP_SCORE = 0
C_SCORE = 1
SCORE = C_SCORE

# Protein-Matrices to be used for Multiple-Protein-Sequence-Alignments

# letters from alphabet not contained in the one-letter-amino-acid-code
other = ["B", "J", "O", "U", "X", "Z"]

# checksum = -204 of the diagonalized matrix
PAM250_LO_MATRIX = [
    # A   B   C   D   E   F   G   H   I  J   K   L   M   N  O   P   Q   R   S   T  U   V   W  X   Y   Z
    [2, 0, -2, 1, 0, -4, 1, -1, -1, 0, -1, -2, -1, 0, 0, 1, 0, -2, 1, 1, 0, 0, -6, 0, -3, 0],  # A x
    [0, 2, -4, 3, 2, -5, 0, 1, -2, 0, 1, -3, -2, 2, 0, -1, 1, -1, 0, 0, 0, -2, -5, 0, -3, 2],  # B x
    [-2, -4, 12, -5, -5, -4, -3, -3, -2, 0, -5, -6, -5, -4, 0, -3, -5, -4, 0, -2, 0, -2, -8, 0, 0, -5],  # C x
    [1, 3, -5, 4, 0, -6, 2, 1, -2, 0, 0, -4, -3, 2, 0, -1, 2, -1, 0, 0, 0, -2, -7, 0, -4, 3],  # D x
    [0, 2, -5, 0, 4, -5, 0, 1, -2, 0, 0, -3, -2, 1, 0, -1, 2, -1, 0, 0, 0, -2, -7, 0, -4, 3],  # E x
    [-4, -5, -4, -6, -5, 9, -5, -2, 1, 0, -5, 2, 0, -4, 0, -5, -5, -4, -3, -3, 0, -1, 0, 0, 7, -5],  # F x
    [1, 0, -3, 2, 0, -5, 5, -2, -3, 0, -2, -4, -3, 0, 0, -1, -1, -3, 1, 0, 0, -1, -7, 0, -5, -1],  # G x
    [-1, 1, -3, 1, 1, -2, -2, 6, -2, 0, 0, -2, -2, 2, 0, 0, 3, 2, -1, -1, 0, -2, -3, 0, 0, 2],  # H x
    [-1, -2, -2, -2, -2, 1, -3, -2, 5, 0, -2, 2, 2, -2, 0, -2, -2, -2, -1, 0, 0, 4, -5, 0, -1, -2],  # I x
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # J x
    [-1, 1, -5, 0, 0, -5, -2, 0, -2, 0, 5, -3, 0, 1, 0, -1, 1, 3, 0, 0, 0, -2, -3, 0, -4, 0],  # K x
    [-2, -3, -6, -4, -3, 2, -4, -2, 2, 0, -3, 6, 4, -3, 0, -3, -2, -3, -3, -2, 0, 2, -2, 0, -1, -3],  # L x
    [-1, -2, -5, -3, -2, 0, -3, -2, 2, 0, 0, 4, 6, -2, 0, -2, -1, 0, -2, -1, 0, -2, -4, 0, -2, -2],  # M x
    [0, 2, -4, 2, 1, -4, 0, 2, -2, 0, 1, -3, -2, 2, 0, -1, 1, 0, 1, 0, 0, -2, -4, 0, -2, 1],  # N x
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O x
    [1, -1, -3, -1, -1, -5, -1, 0, -2, 0, -1, -3, -2, -1, 0, 6, 0, 0, 1, 0, 0, -1, -6, 0, -5, 0],  # P x
    [0, 1, -5, 2, 2, -5, -1, 3, -2, 0, 1, -2, -1, 1, 0, 0, 4, 1, -1, -1, 0, -2, -5, 0, -4, 3],  # Q x
    [-2, -1, -4, -1, -1, -4, -3, 2, -2, 0, 3, -3, 0, 0, 0, 0, 1, 6, 0, -1, 0, -2, 2, 0, -4, 0],  # R x
    [1, 0, 0, 0, 0, -3, 1, -1, -1, 0, 0, -3, -2, 1, 0, 1, -1, 0, 2, 1, 0, -1, -2, 0, -3, 0],  # S x
    [1, 0, -2, 0, 0, -3, 0, -1, 0, 0, 0, -2, -1, 0, 0, 0, -1, -1, 1, 3, 0, 0, -5, 0, -3, -1],  # T x
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U x
    [0, -2, -2, -2, -2, -1, -1, -2, 4, 0, -2, 2, -2, -2, 0, -1, -2, -2, -1, 0, 0, 4, -6, 0, -2, -2],  # V x
    [-6, -5, -8, -7, -7, 0, -7, -3, -5, 0, -3, -2, -4, -4, 0, -6, -5, 2, -2, -5, 0, -6, 17, 0, 0, -6],  # W x
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X x
    [-3, -3, 0, -4, -4, 7, -5, 0, -1, 0, -4, -1, -2, -2, 0, -5, -4, -4, -3, -3, 0, -2, 0, 0, 10, 0],  # Y x
    [0, 2, -5, 3, 3, -5, -1, 2, -2, 0, 0, -3, -2, 1, 0, 0, 3, 0, 0, -1, 0, -2, -6, 0, 0, 3],  # Z x
    # A   B   C   D   E   F   G   H   I  J   K   L   M   N  O   P   Q   R   S   T  U   V   W  X   Y   Z
]
PEPTIDE_SCORE_MATRIX = PAM250_LO_MATRIX

'''
These sequences can be used to test the behaviour of the algorithm.
The sequences are created from the first sequence by replacing or deleting some of the bases
'''
sequences = [
    "ctattgtggcggctgaccggtgactctggaaccctgtagattcgatacccgcgcccgagcatgtggacgtccaggatagtgatcacctcacctagaagca",
    "ctaccgtggcgtgaccggttactctggaccctgtagagcgttaccccgccgagatgtgacgccaggtagtgtgacctcaccatagca",
    "ctattgtgccggctgacggtgactctggaaccggaagattgatacccgccccgagcagtggacgtccagatggtgatccctcacccagggca",
    "ctattgtgggctaccctgactctggaaccctgtagatacgtacccgcacccgagcaagtgggcgtcaggatagtgatcacctacctagaagca",
    "ctattgttgcgctaaccggtgactaagaacccgtagattcgatacccgcgccgagcagggacgtccaggacagtgatcacctacctagaagca",
    "ctatgtggcggctgaccggtgactctggaccctgacattcgtacccgcggccgagctgggacgccaggatatgatcacctcacgtagaaca",
    "ctattatagggctaccgtgactctaaaccctggagattgataccgcgcccgagcatgtggacgtccacgataggatcccctcactagtagc",
    "ctattgtagcgctgaccggggactctagaccctgtagatcgataccgcgcccgacatgtgacgtccaggatagtgacacctcacctttaacc",
    "ctattgggcggctgaacggtgatctggaactctgagatcgaacccgcgaccggcaagtgatgcccaggatactgacccgactgaagca",
    "ctattgtggcggcagacggtgcctctggaacctgtagattcgataccccgcgaaaatgtgacgctaagatgcgatcacccacctagagca",
    "ctatgtggcagctgaccggtgcatggaacctgaaaatcgtaccgcgccgagcagtgcgtccaggatagtaatcaccacctagaaaca",
    "ctttggtggcggctgaccggtgacctggaaccctgtacatcgatacccggcccgcagtgacgaccaggatagtgtacctcactagaagca",
    "ctattgtggctgtgacgcttactctggaaccctgtagattgatcccgcgcccgagcatgtggacgtcctggatagtgatcacctcacctagagca",
    "ctaggtgtcgctgaccggtgactctggaaacctgtagattcgataccccccccgagctgtggcgtccaggaagtatcactcactagaagta",
    "ctataatggcggcgaccggtgacgctggacccctgtgatacgatcccggccccaccaatgacgtccaggatctgatcacctcacaagaaaaa",
    "ctatgtggcgcctgaccggtgactctggaaccctgtagatcgaacccgcccgagcatgtggacgtccggatagtgaccacctcacctaaagca",
    "ctattgtggcagctgacggtgctctgaaaccctgtagattcgatgcctcgcccgacatgtgcacgtcaggatagtgatctcacctatgaga",
    "ctattgtggcgggtgaccggtgctctggacgcgtgagtcgatcccgcgccctgcatgtggcgtcaggagatgatcacctgacctagaagg",
    "tattgtcgcggctgacggtgctctggactctgtagatccgataccccgcccgagctgtggacgtcaggatagagccacctcaactagaaaca",
    "tatatgcggctaccggtgctctcgaaccctgtagattgatacccgcgcccgcgctgtggagccaggatagtgatactcgatagaaggc",
    "cattgtggggctgaccggtgaccctggaaccctgagagtcgataccgcccgacatgggacgcccaggataggattcgccctagaggc",
    "ctttgtgccggctaccgtgactctgaaccctgtagatcgatccccggcccgagcatgtggacgttaggatagtcatcccctaccgaaagca",
    "ctattgggggctgacctgtgactctggaagcctgaactcatacccgagcacgagcatggcacgtccaagatagtgatcacctcacgagaagca",
    "ctatttggcggctgaccggtgactctggaaccctgagttcgatacccgcgccgtgcatggcgactccaggatatgatcaccctcctagaagca",
    "cgactgtggcggctgaacgtgactcgggaaccctatagatcgatccccgcgtccgacatgggacgtcctggatagtggaccccacctgaagca",
    "ctatgtggggctgagcgtgactctggaacccgtagattcatcccgcacccggatgtgtcctccaggataggatcacctcacctgaagca"]

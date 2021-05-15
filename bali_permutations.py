import os

from utils.utils import read_fasta_data


def permutation(original, test):
    assert (set(original) == set(test) and len(original) == len(test))
    permutation = []
    for element in original:
        permutation.append(test.index(element))
    return permutation


bali_dir = "../BAliBASE3/"
for value_file in os.listdir("C:/Users/joere/Desktop/alignments/BAliBASE/"):
    if "Value" not in value_file:
        continue
    _, header = read_fasta_data(os.path.join("C:/Users/joere/Desktop/alignments/BAliBASE/", value_file))
    bb_code = value_file[-11:-4]
    rv_code = bb_code[2:4]
    name = value_file[2:-12]
    bali_file = os.path.join(bali_dir, "RV" + rv_code, bb_code + ".tfa")
    _, bali_header = read_fasta_data(bali_file)
    print(name + ": RV" + rv_code + "/" + bb_code + ".tfa:", permutation(bali_header, header))

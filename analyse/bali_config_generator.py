import pandas as pd

# hyperparameter to restrict the analysed sequences
count_min_cutoff = 3
length_min_cutoff = 0
count_max_cutoff = 10
length_max_cutoff = 500

console_output = True
file_output = False

# take the data stored in the comparison to not open each sequence file unnecessarily often
df = pd.read_csv("balibase_comparison.csv", header=0)

# write the configuration file
if file_output:
    output = open("./bali_config.json", "w")
    output.write("{\n\t\"Benchmarks\": [")

# write the benchmark-configuration to the file ...
for index, row in df.iterrows():
    # ... if the file fulfills the requirements ...
    if count_min_cutoff <= int(row["nseqs"]) <= count_max_cutoff and \
            length_min_cutoff <= int(row["lseqs"]) <= length_max_cutoff:
        # print("\"" + row["name"] + "\",")
        print(row["name"], ":\t", row["nseqs"], "|", row["lseqs"])
        if file_output:
            output.write("{\n\t\t\"Name\": \"../BAliBASE3/" + row["name"] + ".tfa\",\n\t\t\"IDs\":[0]\n\t}, ")

# ... and finally close the file
if file_output:
    output.write("]\n}\n")
    output.flush()
    output.close()

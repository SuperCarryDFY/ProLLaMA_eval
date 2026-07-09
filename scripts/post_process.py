import json 
## extract text and seqs to jsonl and fasta
res = []
with open("scripts/output.txt", 'r') as f:
    for line in f:
        all_splits = []
        splits = line.strip().split("<")
        for split in splits:
            all_splits.extend(split.split(">"))
        try:
            text = all_splits[1]
            seq = all_splits[3]
            res.append((text, seq))
        except:
            continue
            # print(line)


with open("scripts/seqs.fasta", 'w' ) as f:
    for idx, (text, seq) in enumerate(res):
        f.write(f">idx_{idx}\n{seq}\n")

with open("scripts/res.jsonl", "w") as f:
    for idx, (text, seq) in enumerate(res):
        json.dump({"idx": idx, "text": text, "seq": seq}, f)
        f.write("\n")
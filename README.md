## Running ProLLaMA

First, clone the official ProLLaMA repository:

```bash
git clone git@github.com:SuperCarryDFY/ProLLaMA_eval.git
cd ProLLaMA_eval/src; git clone git@github.com:PKU-YuanGroup/ProLLaMA.git
cd ..
```

Prepare the ProLLaMA checkpoint under `checkpoints/`. The inference script uses
`src/ProLLaMA/scripts/infer.py`, and the default model path is `./checkpoints`.

The full evaluation script runs three steps:

1. ProLLaMA sequence generation.
2. Post-processing into `scripts/res.jsonl` and `scripts/seqs.fasta`.
3. ESMFold pLDDT evaluation and ProTrek score evaluation.

Before running ProTrek scoring, set the following environment variables:

```bash
export PROTREKCODEPATH=/path/to/ProTrek/code
export PROTREKWEIGHT=/path/to/ProTrekWeight
```

The ESMFold step uses Ray for multi-GPU inference, so make sure `ray` is installed
in the environment used to run the script. The ESMFold step also requires `esm`,
`biotite`, `torch`, `numpy`, and `tqdm`.

Then run ProLLaMA inference and evaluation from the repository root:

```bash
bash scripts/infer_and_eval.sh
```

The main outputs are:

- `scripts/output.txt`: raw ProLLaMA generations
- `scripts/res.jsonl`: parsed text-sequence pairs
- `scripts/res_protrek_score.json`: ProTrek scores
- `scripts/seqs_esmfold_results.json`: ESMFold pLDDT, pTM, and PAE metrics

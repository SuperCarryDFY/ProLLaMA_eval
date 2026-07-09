: "${PROTREKCODEPATH:?Please set PROTREKCODEPATH to the ProTrek code directory.}"
: "${PROTREKWEIGHT:?Please set PROTREKWEIGHT to the ProTrek weight directory.}"

python src/ProLLaMA/scripts/infer.py --input_file scripts/input.txt --output_file scripts/output.txt

# post-process and eval
python scripts/post_process.py
python scripts/esmfold_ray.py --fasta_path scripts/seqs.fasta

python scripts/protrek_score.py scripts/res.jsonl

import argparse
import json
import logging
import os

import biotite.structure.io as bsio
import esm
import numpy as np
import ray
import torch
from collections import defaultdict
from ray.util.actor_pool import ActorPool
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [ESMFold] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def get_pLDDT(pdb_file):
    struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    return struct.b_factor.mean()


def calculate_pae(out):
    pae = (out["aligned_confidence_probs"][0].cpu().numpy() * np.arange(64)).mean(
        -1
    ) * 31
    mask = out["atom37_atom_exists"][0, :, 1] == 1
    mask = mask.cpu()
    pae = pae[mask, :][:, mask]
    return np.mean(pae)


def normalize_sequence(sequence):
    return "".join(sequence.split()).upper()


def validate_sequence(entry_id, sequence):
    sequence = normalize_sequence(sequence)
    if not sequence:
        return None, "empty sequence"
    if not (set(sequence) & STANDARD_AMINO_ACIDS):
        return None, "sequence contains no standard amino acid tokens"
    return sequence, None


@ray.remote(num_gpus=1)
class ESMFoldPredictor:
    def __init__(self, cache_dir):
        os.environ["TORCH_HOME"] = cache_dir

        logger.info("Loading ESMFold model on GPU...")
        self.model = esm.pretrained.esmfold_v1().eval().cuda()
        logger.info("Model loaded successfully.")

    def process_batch(self, batch_data, output_path):
        """
        处理一个小批次的数据
        batch_data: list of (entry_id, sequence)
        """
        batch_metrics_dic = defaultdict(dict)
        batch_failed_dic = {}

        for entry_id, sequence in batch_data:
            sequence = sequence[:1024]  # 截断
            output_file = f"sequence_{entry_id}.pdb"
            file_full_path = os.path.join(output_path, output_file)

            try:
                with torch.no_grad():
                    out = self.model.infer([sequence])
                    pdb_out = self.model.output_to_pdb(out)[0]

                pae_val = calculate_pae(out)
                batch_metrics_dic[output_file]["plddt"] = out["mean_plddt"][0].item()
                batch_metrics_dic[output_file]["ptm"] = out["ptm"][0].item()
                batch_metrics_dic[output_file]["pae"] = pae_val

                with open(file_full_path, "w") as f:
                    f.write(pdb_out)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"Error predicting {entry_id}: {error_msg}")
                batch_failed_dic[entry_id] = {
                    "sequence_length": len(sequence),
                    "error": error_msg,
                }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"Error predicting {entry_id}: {error_msg}")
                batch_failed_dic[entry_id] = {
                    "sequence_length": len(sequence),
                    "error": error_msg,
                }

        return dict(batch_metrics_dic), batch_failed_dic


def read_fasta(file_path, max_seqs=1000000):
    sequences = {}
    current_header = None
    current_sequence = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    # 保存上一个序列
                    sequences[current_header] = "".join(current_sequence)
                    current_sequence = []
                current_header = line[1:]  # 去掉'>'符号
            else:
                current_sequence.append(line)
            if len(sequences) > max_seqs:
                break
        # 保存最后一个序列
        if current_header is not None:
            sequences[current_header] = "".join(current_sequence)

    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    cache_dir = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    output_path = args.fasta_path.replace(".fasta", "_esmfold_results")
    os.makedirs(output_path, exist_ok=True)

    ray_tmpdir = os.environ.get("RAY_TMPDIR", f"/tmp/ray_{os.getpid()}")
    ray.init(
        address="local",
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=ray_tmpdir,
    )

    sequences_dict = read_fasta(args.fasta_path)
    logger.info(f"Reading sequences from {args.fasta_path}")

    all_items = []
    skipped_sequences_dic = {}
    for name, seq in sequences_dict.items():
        sequence_length = len(normalize_sequence(seq))
        seq, error_msg = validate_sequence(name, seq)
        if error_msg is not None:
            skipped_sequences_dic[name] = {
                "sequence_length": sequence_length,
                "error": error_msg,
            }
            continue
        all_items.append((name, seq))

    logger.info(f"Total sequences: {len(all_items)}")
    if skipped_sequences_dic:
        logger.warning(f"Skipped {len(skipped_sequences_dic)} invalid sequences before inference.")

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("ESMFold Ray evaluation requires at least one CUDA GPU.")
    logger.info(f"Detected {num_gpus} GPUs. Spawning actors...")

    actors = [ESMFoldPredictor.remote(cache_dir) for _ in range(num_gpus)]
    pool = ActorPool(actors)

    BATCH_SIZE = max(int(args.batch_size), 1)
    chunks = [
        all_items[i : i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)
    ]

    logger.info(
        f"Split data into {len(chunks)} batches (size={BATCH_SIZE}). Starting inference..."
    )

    final_batch_metrics_dic = {}
    failed_sequences_dic = dict(skipped_sequences_dic)

    pbar = tqdm(total=len(chunks), desc="Processing Batches", unit="batch")

    for batch_metrics_dic, batch_failed_dic in pool.map_unordered(
        lambda actor, value: actor.process_batch.remote(value, output_path), chunks
    ):
        final_batch_metrics_dic.update(batch_metrics_dic)
        failed_sequences_dic.update(batch_failed_dic)
        pbar.update(1)

    pbar.close()

    logger.info("Inference done. Calculating metrics...")
    if final_batch_metrics_dic:
        mean_pae = np.mean([v["pae"] for v in final_batch_metrics_dic.values()])
        mean_plddt = np.mean([v["plddt"] for v in final_batch_metrics_dic.values()])
        mean_ptm = np.mean([v["ptm"] for v in final_batch_metrics_dic.values()])
    else:
        mean_pae = None
        mean_plddt = None
        mean_ptm = None

    logger.info(f"Successful predictions: {len(final_batch_metrics_dic)}")
    logger.info(f"Failed or skipped predictions: {len(failed_sequences_dic)}")
    if final_batch_metrics_dic:
        logger.info(f"Mean pLDDT: {mean_plddt:.4f}")
        logger.info(f"Mean PAE: {mean_pae:.4f}")
        logger.info(f"Mean pTM: {mean_ptm:.4f}")

    results_path = args.fasta_path.replace(".fasta", "_esmfold_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            result_dic = json.load(f)
    else:
        result_dic = {}

    result_dic["ESMFold pLDDT"] = float(mean_plddt) if mean_plddt is not None else None
    result_dic["ESMFold pae"] = float(mean_pae) if mean_pae is not None else None
    result_dic["ESMFold pTM"] = float(mean_ptm) if mean_ptm is not None else None
    result_dic["ESMFold successful_predictions"] = len(final_batch_metrics_dic)
    result_dic["ESMFold failed_predictions"] = len(failed_sequences_dic)
    for k, v in final_batch_metrics_dic.items():
        result_dic[f"{k}_metrics"] = v
    if failed_sequences_dic:
        result_dic["ESMFold failed_sequences"] = failed_sequences_dic

    with open(results_path, "w") as f:
        json.dump(result_dic, f, indent=4)

    logger.info(f"Metrics saved to {results_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()

import sys

sys.path.append("..")  # Ensure src is in the path
import argparse
from pathlib import Path

from src.infra.gym.checkpoints import LocalCheckpointStorageAdapter
from src.infra.gym.model import HFModelRepository
from src.infra.gld.prof_oak_pc import LocalProfOakPcRepository
from src.domain.gym.model import ModelCard


def main(args):
    repo = HFModelRepository()

    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = LocalCheckpointStorageAdapter(
            base_path=args.checkpoint_base_path,
        ).get_latest_checkpoint()

    if checkpoint_path is None:
        raise RuntimeError(
            f"No checkpoint found under {args.checkpoint_base_path}. "
            "Use --checkpoint-path to specify one explicitly.",
        )

    print(f"Uploading checkpoint: {checkpoint_path}")

    model_card = (
        ModelCard(
            version=args.version,
            dataset_version=args.dataset_repo_id or args.dataset_partition,
            context_length=args.context_length,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            description=args.description,
        )
        if any(
            [
                args.version,
                args.dataset_repo_id,
                args.dataset_partition,
                args.description,
                args.context_length,
                args.n_embd,
                args.n_layer,
                args.n_head,
            ],
        )
        else None
    )

    model_url = repo.upload(
        checkpoint_path=checkpoint_path,
        repo_id=args.repo_id,
        version=args.version,
        model_card=model_card,
        model_code_path=args.model_code_path,
    )
    print(f"Model uploaded to: {model_url}")

    if args.dataset_repo_id:
        print(f"Uploading dataset (partition: {args.dataset_partition})...")
        box_entity = LocalProfOakPcRepository(
            base_path=args.data_base_path,
            partition=args.dataset_partition,
        ).load()
        dataset_url = repo.upload_dataset(
            box_entity=box_entity,
            repo_id=args.dataset_repo_id,
        )
        print(f"Dataset uploaded to: {dataset_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a checkpoint to HuggingFace Hub",
    )

    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo id, e.g. iamthinbaker/GPokeT2",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the checkpoint directory. Defaults to latest found under --checkpoint-base-path.",
    )
    parser.add_argument(
        "--checkpoint-base-path",
        type=str,
        default="/workspace",
        help="Base path to search for the latest checkpoint. Default: /workspace",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version tag, e.g. v0.1-wip",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Free-text description for the model card",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-embd",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-layer",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo id. If provided, uploads the dataset too.",
    )
    parser.add_argument(
        "--dataset-partition",
        type=str,
        default="latest",
        help="Dataset partition to upload. Default: latest",
    )
    parser.add_argument(
        "--data-base-path",
        type=str,
        default="/workspace",
        help="Base path for the local dataset. Default: /workspace",
    )
    parser.add_argument(
        "--model-code-path",
        type=str,
        default=None,
        help="Path to modeling_conditioned_gpt2.py para habilitar trust_remote_code en HF.",
    )

    main(parser.parse_args())

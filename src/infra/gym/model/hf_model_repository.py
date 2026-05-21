import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from safetensors import safe_open

from src.domain.gld.prof_oak_pc import BoxEntity
from src.domain.gym.model import ModelCard, ModelRepository


class HFModelRepository(ModelRepository):
    def __init__(
        self,
        token: str | None = None,
    ):
        self.token = token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def upload(
        self,
        checkpoint_path: Path | str,
        repo_id: str,
        version: str | None = None,
        model_card: ModelCard | None = None,
        model_code_path: Path | str | None = None,
    ) -> str:
        checkpoint_path = Path(checkpoint_path)

        self.api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )

        self.api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload model {version}"
            if version
            else "Upload model",
        )

        if model_code_path is not None:
            self._push_model_code(
                repo_id,
                Path(model_code_path),
                checkpoint_path,
            )

        if model_card is not None:
            self._push_model_card(repo_id, model_card)

        if version is not None:
            try:
                self.api.delete_tag(
                    repo_id=repo_id,
                    tag=version,
                    repo_type="model",
                )
            except Exception:
                pass
            self.api.create_tag(repo_id=repo_id, tag=version, repo_type="model")

        return f"https://huggingface.co/{repo_id}"

    def _push_model_code(
        self,
        repo_id: str,
        model_code_path: Path,
        checkpoint_path: Path,
    ) -> None:
        module_name = model_code_path.stem

        self.api.upload_file(
            path_or_fileobj=str(model_code_path),
            path_in_repo=model_code_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model code for trust_remote_code",
        )

        config_path = checkpoint_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        config["auto_map"] = {
            "AutoModelForCausalLM": f"{module_name}.ConditionedGPT2",
        }

        if "num_pokemon" not in config:
            config["num_pokemon"] = self._read_num_pokemon(checkpoint_path)

        self.api.upload_file(
            path_or_fileobj=json.dumps(config, indent=2).encode(),
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add auto_map for trust_remote_code",
        )

    def upload_dataset(
        self,
        box_entity: BoxEntity,
        repo_id: str,
    ) -> str:
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

        box_entity.dataset.push_to_hub(repo_id, token=self.token)

        with tempfile.TemporaryDirectory() as tmp:
            box_entity.tokenizer.save_pretrained(tmp)
            self.api.upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Add tokenizer",
            )

        self._push_dataset_card(repo_id)

        return f"https://huggingface.co/datasets/{repo_id}"

    def _push_dataset_card(self, repo_id: str) -> None:
        card_path = Path(__file__).parents[4] / "DATASET_CARD.md"
        readme = card_path.read_text().replace("{repo_id}", repo_id)

        self.api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )

    def _push_model_card(self, repo_id: str, card: ModelCard) -> None:
        readme = (Path(__file__).parents[4] / "README.md").read_text()
        readme = readme.replace("{repo_id}", repo_id)

        self.api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

    def _read_num_pokemon(self, checkpoint_path: Path) -> int:
        safetensors_path = checkpoint_path / "model.safetensors"
        with safe_open(str(safetensors_path), framework="pt") as f:
            return f.get_slice("conditioning.weight").get_shape()[0]

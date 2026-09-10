"""Upload a trained CellposeSAM model checkpoint to the Hugging Face Hub.

Setup (once):
    pip install huggingface_hub
    huggingface-cli login          # or: export HF_TOKEN=...
    edit REPO_ID in hf_config.py to your own namespace

Usage:
    python scripts/upload_model.py <local_model_path>
    python scripts/upload_model.py <local_model_path> --repo-id you/your-model --private
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

from hf_config import REPO_ID


MODEL_CARD_TEMPLATE = """\
# {repo_name}

A [CellposeSAM](https://github.com/MouseLand/cellpose) model fine-tuned for \
Purkinje cell segmentation in H&E-stained brain tissue (20x magnification).

Trained starting from the `cpsam_v2` base model via human-in-the-loop \
correction in the Cellpose GUI.

## Usage

```python
from huggingface_hub import hf_hub_download
from cellpose import models

model_path = hf_hub_download(repo_id="{repo_id}", filename="{filename}")
model = models.CellposeModel(gpu=True, pretrained_model=model_path)
masks, flows, styles = model.eval(image, diameter=None)
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Local path to the trained model checkpoint")
    parser.add_argument("--repo-id", default=REPO_ID,
                         help=f"Hugging Face model repo id. Defaults to {REPO_ID!r} "
                              "(edit scripts/hf_config.py to change the default).")
    parser.add_argument("--private", action="store_true",
                         help="Create/keep the repo private (default: public)")
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    if args.repo_id == "<your-hf-username>/purkinje-cellpose-sam":
        parser.error("Set REPO_ID in scripts/hf_config.py (or pass --repo-id) to your own namespace first.")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    api = HfApi()
    already_existed = api.repo_exists(repo_id=args.repo_id, repo_type="model")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading {model_path} -> {args.repo_id}")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message or f"Upload {model_path.name}",
    )

    if not already_existed:
        repo_name = args.repo_id.split("/")[-1]
        card = MODEL_CARD_TEMPLATE.format(repo_name=repo_name, repo_id=args.repo_id,
                                           filename=model_path.name)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add model card",
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

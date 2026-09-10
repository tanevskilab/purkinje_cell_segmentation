"""Download a CellposeSAM model checkpoint from the Hugging Face Hub.

Setup (once):
    pip install huggingface_hub
    # Only needed for a private repo:
    huggingface-cli login          # or: export HF_TOKEN=...

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --repo-id you/your-model --output-dir models/
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from hf_config import REPO_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=REPO_ID,
                         help=f"Hugging Face model repo id. Defaults to {REPO_ID!r} "
                              "(edit scripts/hf_config.py to change the default).")
    parser.add_argument("--filename", default=None,
                         help="Specific file to download. Defaults to the only/first "
                              "non-README file in the repo.")
    parser.add_argument("--output-dir", default="models",
                         help="Local directory to place the downloaded model in. Defaults to 'models/'.")
    args = parser.parse_args()

    if args.repo_id == "<your-hf-username>/purkinje-cellpose-sam":
        parser.error("Set REPO_ID in scripts/hf_config.py (or pass --repo-id) to the model's namespace first.")

    filename = args.filename
    if filename is None:
        skip = {"README.md", ".gitattributes"}
        files = [f for f in list_repo_files(args.repo_id, repo_type="model") if f not in skip]
        if not files:
            raise FileNotFoundError(f"No model files found in {args.repo_id}")
        if len(files) > 1:
            print(f"Multiple files found in {args.repo_id}: {files}\nDownloading the first: {files[0]}")
        filename = files[0]

    print(f"Downloading {args.repo_id}/{filename} ...")
    local_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        repo_type="model",
        local_dir=args.output_dir,
    )
    print(f"Saved to {local_path}")


if __name__ == "__main__":
    main()

"""
HuggingFace Hub utilities for pushing and loading model checkpoints.

Models are saved as plain PyTorch state_dicts together with a small
metadata JSON (vocab, model config) so they can be fully reconstructed
from the Hub.

Usage:
    push_model(model, vocab_meta, repo_id="username/model-name")
    state_dict, meta = pull_model(repo_id="username/model-name")
"""

import json
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def push_model(
    model: nn.Module,
    vocab_meta: Dict[str, Any],
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    token: Optional[str] = None,
    private: bool = False,
) -> None:
    """
    Push a PyTorch model (state_dict + meta) to the HuggingFace Hub.

    Args:
        model:          trained model
        vocab_meta:     dict with vocab, config, and any other metadata to store
        repo_id:        Hub repo id, e.g. "username/mymodel"
        model_filename: filename for the weights inside the repo
        token:          HF auth token (falls back to HF_TOKEN env var)
        private:        create a private repository
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[HF] huggingface_hub not installed – push skipped.")
        return

    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        print("[HF] No HuggingFace token found. Set HF_TOKEN env var or pass token=.")
        return

    api = HfApi(token=token)

    # Create repo if it does not exist
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        print(f"[HF] Could not create repo: {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save weights
        weights_path = os.path.join(tmpdir, model_filename)
        torch.save(model.state_dict(), weights_path)

        # Save metadata
        meta_path = os.path.join(tmpdir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(vocab_meta, f, indent=2)

        # Upload both files
        api.upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            repo_type="model",
        )

    print(f"[HF] Model pushed to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def pull_model(
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    token: Optional[str] = None,
    map_location: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Download a model state_dict and metadata from the HuggingFace Hub.

    Args:
        repo_id:        Hub repo id
        model_filename: filename of the weights inside the repo
        token:          HF auth token (falls back to HF_TOKEN env var)
        map_location:   torch map_location for loading state_dict

    Returns:
        (state_dict, metadata_dict)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required: pip install huggingface_hub")

    token = token or os.environ.get("HF_TOKEN")

    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        token=token,
    )
    meta_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta.json",
        token=token,
    )

    state_dict = torch.load(weights_path, map_location=map_location)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"[HF] Model loaded from https://huggingface.co/{repo_id}")
    return state_dict, meta


# ---------------------------------------------------------------------------
# Local checkpoint helpers (alternative to HF)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    vocab_meta: Dict[str, Any],
    path: str,
) -> None:
    """Save model + metadata to a local file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": vocab_meta}, path)
    print(f"[Checkpoint] Saved → {path}")


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load model state_dict and metadata from a local file."""
    ckpt = torch.load(path, map_location=map_location)
    return ckpt["state_dict"], ckpt["meta"]

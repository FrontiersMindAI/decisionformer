"""Hugging Face Hub uploads for decision checkpoints."""

from __future__ import annotations

from pathlib import Path

PLACEHOLDER = "<this repo>"


def push_folder(directory: str | Path, repo_id: str, *, private: bool = True, token: str | None = None,
                commit_message: str = "Upload decision model") -> str:
    """Upload a checkpoint directory (the standard layout, with its model card) to `repo_id`.

    The model card's `<this repo>` placeholders are filled with `repo_id` in the upload; the local copy is unchanged.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    folder = Path(directory)
    if not (folder / "decision_config.json").is_file():
        raise ValueError(f"{folder} is not a decision checkpoint")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    card = folder / "README.md"
    api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message=commit_message,
                      ignore_patterns=[".cache/*", "*.tmp", *(["README.md"] if card.is_file() else [])])
    if card.is_file():
        text = card.read_text().replace(PLACEHOLDER, repo_id)
        api.create_commit(repo_id=repo_id, commit_message="Add model card",
                          operations=[CommitOperationAdd("README.md", text.encode())])
    return f"https://huggingface.co/{repo_id}"

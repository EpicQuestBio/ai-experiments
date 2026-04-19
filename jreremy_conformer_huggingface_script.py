# /// script
# dependencies = [
#   "huggingface_hub",
#   "torch==2.6",
#   "torchaudio==2.6.0",
#   "numpy",
#   "torchmetrics",
#   "torchcodec==0.2"
# ]
# ///

from pathlib import Path
import os
import subprocess
import sys

from huggingface_hub import HfApi


ROOT = Path.cwd()
WORKDIR = ROOT / "work"
REPO_DIR = WORKDIR / "conformer"

HF_NAMESPACE = os.environ.get("HF_NAMESPACE")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "conformer-librispeech-test")
HF_TOKEN = os.environ.get("HF_TOKEN")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    WORKDIR.mkdir(parents=True, exist_ok=True)

    # Make sure you have the right version of ffmpeg (4/5/6) for TorchAudio
    run(["cat", "/etc/os-release"])
    run(["apt", "update"])
    run(["apt", "install", "-y", "ffmpeg"])
    run(["ffmpeg", "-version"])

    if not REPO_DIR.exists():
        run(["git", "clone", "https://github.com/jreremy/conformer.git", str(REPO_DIR)])

    train_py = REPO_DIR / "train.py"
    text = train_py.read_text(encoding="utf-8")
    text = text.replace(
        "parser.add_argument('--smart_batch', type=bool, default=True, help='Use smart batching for faster training')",
        "parser.add_argument('--smart_batch', type=bool, default=False, help='Use smart batching for faster training')"
    )
    train_py.write_text(text, encoding="utf-8")
    print("Patched train.py: smart_batch default set to False", flush=True)

    # It downloads the dataset automatically.
    print("Running Conformer", flush=True)
    run([
        sys.executable,
        "-u",
        "train.py",
        "--data_dir=./data",
        "--train_set=train-clean-100",
        "--test_set=test-clean",
        "--checkpoint_path=model_best.pt",
        "--epochs=1"
    ], cwd=REPO_DIR)

    ckpt = REPO_DIR / "model_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt}")

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set; cannot upload model.")

    print("Saving model", flush=True)
    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_NAMESPACE}/{HF_MODEL_REPO}"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="model_best.pt",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Uploaded checkpoint to https://huggingface.co/{repo_id}", flush=True)


if __name__ == "__main__":
    main()

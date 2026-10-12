import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from pipeline.weights import MANIFEST, MODEL_NAMES, weights_path

SPACE_ID = "cedmonst22/Catan-Board-Visualizer"
SPACE_FILES = {"space/README.md": "README.md", "space/requirements.txt": "requirements.txt",
               "space/packages.txt": "packages.txt"}
CODE_FILES = ["app.py", "assets/corner_guide.jpg", "data_prep/__init__.py", "data_prep/apply_masks.py",
              "data_prep/click_corners.py", "data_prep/extract_crops.py"]
KEEP_REMOTE = {".gitattributes"}


def files_to_upload() -> dict[str, Path]:
    files = {name: Path(name) for name in CODE_FILES}
    files.update({path.as_posix(): path for path in sorted(Path("pipeline").glob("*.py"))})
    files[MANIFEST.as_posix()] = MANIFEST
    files.update({weights_path(name).as_posix(): weights_path(name) for name in MODEL_NAMES})
    files.update({remote: Path(local) for local, remote in SPACE_FILES.items()})
    return files


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def problems(files: dict[str, Path]) -> list[str]:
    found = [f"missing {path}" for path in files.values() if not path.is_file()]
    tracked = [path.as_posix() for path in files.values() if path.suffix != ".pt"]
    dirty = git("status", "--porcelain", "--", *tracked)
    if dirty:
        found.append(f"uncommitted changes, commit them first:\n{dirty}")
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    for name in MODEL_NAMES:
        path = weights_path(name)
        expected = manifest.get(name, {}).get("sha256")
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            found.append(f"{path} does not match {MANIFEST}, promote it again")
    return found


def stage(files: dict[str, Path], folder: Path) -> None:
    for remote, local in files.items():
        target = folder / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
    print(f"staged {len(files)} files in {folder}")


def deploy(files: dict[str, Path]) -> None:
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as error:
        print(f"not logged in to Hugging Face ({error}), run: .venv\\Scripts\\hf auth login")
        sys.exit(1)
    remote_files = set(api.list_repo_files(SPACE_ID, repo_type="space"))
    stale = sorted(remote_files - set(files) - KEEP_REMOTE)
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)) for remote, local in files.items()]
    operations += [CommitOperationDelete(path_in_repo=path) for path in stale]
    message = f"deploy {git('rev-parse', '--short', 'HEAD')}"
    commit = api.create_commit(SPACE_ID, operations, commit_message=message, repo_type="space")
    print(f"{user} uploaded {len(files)} files, removed {len(stale)} stale ({', '.join(stale) or 'none'})")
    print(f"{message}: {commit.commit_url}")
    print(f"build logs: https://huggingface.co/spaces/{SPACE_ID}?logs=build")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"upload the app to the {SPACE_ID} Space")
    parser.add_argument("--dry-run", action="store_true", help="check and list the files without uploading")
    parser.add_argument("--stage", type=Path, help="copy the files that would be uploaded into this folder")
    args = parser.parse_args()

    files = files_to_upload()
    found = problems(files)
    for problem in found:
        print(problem)
    if args.stage:
        stage(files, args.stage)
    if args.dry_run or args.stage:
        size = sum(path.stat().st_size for path in files.values() if path.is_file())
        print("\n".join(f"  {remote}" for remote in sorted(files)))
        print(f"{len(files)} files, {size / 1e6:.1f} MB, {'not ready' if found else 'ready'} to deploy")
        return
    if found:
        sys.exit(1)
    deploy(files)


if __name__ == "__main__":
    main()

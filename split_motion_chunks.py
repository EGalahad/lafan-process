import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


CHUNK_KEYS = [
    "body_pos_w",
    "body_quat_w",
    "joint_pos",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_vel",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split motion.npz sequences into fixed-length chunks."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing <motion_name>/motion.npz + meta.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for chunked motions.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=1000,
        help="Chunk length in frames.",
    )
    return parser.parse_args()


def _load_motion(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _save_chunk(chunk_dir: Path, motion: dict, meta: dict) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    np.savez(chunk_dir / "motion.npz", **motion)
    (chunk_dir / "meta.json").write_text(json.dumps(meta))


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    motion_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not motion_dirs:
        raise FileNotFoundError(f"No motion folders found in {input_dir}")

    for motion_dir in tqdm(motion_dirs, desc="Motions", unit="motion"):
        motion_path = motion_dir / "motion.npz"
        meta_path = motion_dir / "meta.json"
        if not motion_path.is_file() or not meta_path.is_file():
            continue

        motion = _load_motion(motion_path)
        meta = json.loads(meta_path.read_text())

        total = motion["joint_pos"].shape[0]
        for start in range(0, total, args.chunk_len):
            end = min(start + args.chunk_len, total)
            chunk = {k: motion[k][start:end] for k in CHUNK_KEYS}
            chunk_dir = output_dir / motion_dir.name / f"{start}-{end}"
            _save_chunk(chunk_dir, chunk, meta)


if __name__ == "__main__":
    main()

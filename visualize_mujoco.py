import argparse
import os
import time

import mujoco
import numpy as np
from mujoco import viewer
from tqdm import tqdm


G1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


DEFAULT_G1_MJCF = "g1_xmls/g1.xml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LAFAN1 retargeted G1 motions with MuJoCo."
    )
    parser.add_argument(
        "--csv",
        default="LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv",
        help="CSV path relative to this repo or absolute path.",
    )
    parser.add_argument(
        "--mjcf",
        default=os.environ.get("G1_MJCF_PATH", DEFAULT_G1_MJCF),
        help="Path to the G1 MJCF file.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a viewer window and without sleeping.",
    )
    return parser.parse_args()


def _load_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data


def _apply_frame(model: mujoco.MjModel, data: mujoco.MjData, frame: np.ndarray) -> None:
    if frame.shape[0] != 7 + len(G1_JOINT_ORDER):
        raise ValueError(
            "Expected frame width 36 (7 root + 29 joints), got "
            f"{frame.shape[0]}."
        )

    root_xyz = frame[0:3]
    root_qxqyqzqw = frame[3:7]
    root_qwxyz = np.array(
        [root_qxqyqzqw[3], root_qxqyqzqw[0], root_qxqyqzqw[1], root_qxqyqzqw[2]]
    )

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_adr = model.jnt_qposadr[base_id]
    data.qpos[base_adr : base_adr + 3] = root_xyz
    data.qpos[base_adr + 3 : base_adr + 7] = root_qwxyz

    joint_values = frame[7:]
    for name, value in zip(G1_JOINT_ORDER, joint_values):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jadr = model.jnt_qposadr[jid]
        data.qpos[jadr] = value

    mujoco.mj_forward(model, data)


def main() -> None:
    args = _parse_args()

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not os.path.isfile(args.mjcf):
        raise FileNotFoundError(
            f"MJCF file not found: {args.mjcf}. Set --mjcf or G1_MJCF_PATH."
        )

    motion = _load_csv(csv_path)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    end = args.end if args.end >= 0 else motion.shape[0]
    frame_indices = range(args.start, min(end, motion.shape[0]), max(1, args.stride))

    if args.headless:
        for idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            _apply_frame(model, data, motion[idx])
        return

    frame_dt = 1.0 / args.fps
    next_time = time.time()

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        for idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            if not v.is_running():
                break
            _apply_frame(model, data, motion[idx])
            v.sync()
            next_time += frame_dt * args.stride
            sleep_for = next_time - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    main()

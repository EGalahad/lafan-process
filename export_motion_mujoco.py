import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm


CSV_JOINT_ORDER = [
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
        description="Export LAFAN1 G1 CSV motions into HDMI motion.npz format."
    )
    parser.add_argument(
        "--csv-folder",
        default="LAFAN1_Retargeting_Dataset/g1",
        help="Folder with CSV files (relative to this repo or absolute path).",
    )
    parser.add_argument(
        "--mjcf",
        default=os.environ.get("G1_MJCF_PATH", DEFAULT_G1_MJCF),
        help="Path to the G1 MJCF file.",
    )
    parser.add_argument(
        "--out-dir",
        default="output_motion",
        help="Output directory for per-motion subfolders.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="CSV FPS.")
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    return parser.parse_args()


def _load_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data


def _build_qpos_sequence(
    model: mujoco.MjModel, motion: np.ndarray, joint_names: list[str]
) -> np.ndarray:
    expected = 7 + len(joint_names)
    if motion.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} columns (7 root + {len(joint_names)} joints), "
            f"got {motion.shape[1]}."
        )
    qpos = np.zeros((motion.shape[0], model.nq))
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    base_adr = model.jnt_qposadr[base_id]

    for t in range(motion.shape[0]):
        frame = motion[t]
        root_xyz = frame[0:3]
        root_qxqyqzqw = frame[3:7]
        root_qwxyz = np.array(
            [root_qxqyqzqw[3], root_qxqyqzqw[0], root_qxqyqzqw[1], root_qxqyqzqw[2]]
        )
        qpos[t, base_adr : base_adr + 3] = root_xyz
        qpos[t, base_adr + 3 : base_adr + 7] = root_qwxyz

        joint_values = frame[7:]
        for name, value in zip(joint_names, joint_values):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            jadr = model.jnt_qposadr[jid]
            qpos[t, jadr] = value
    return qpos


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / fps
    qvel = np.zeros((qpos.shape[0], model.nv))
    for t in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(model, qvel[t], dt, qpos[t], qpos[t + 1])
    if qpos.shape[0] > 1:
        qvel[-1] = qvel[-2]
    return qvel


def main() -> None:
    args = _parse_args()

    csv_folder = args.csv_folder
    if not os.path.isabs(csv_folder):
        csv_folder = os.path.join(os.path.dirname(__file__), csv_folder)
    csv_folder = Path(csv_folder)
    if not csv_folder.is_dir():
        raise FileNotFoundError(f"CSV folder not found: {csv_folder}")

    if not os.path.isfile(args.mjcf):
        raise FileNotFoundError(
            f"MJCF file not found: {args.mjcf}. Set --mjcf or G1_MJCF_PATH."
        )

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    body_ids = list(range(model.nbody))
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in body_ids
    ]

    hinge_joint_ids = [
        i
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in hinge_joint_ids
    ]
    joint_qpos_addrs = [model.jnt_qposadr[i] for i in hinge_joint_ids]
    joint_dof_addrs = [model.jnt_dofadr[i] for i in hinge_joint_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(csv_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_folder}")

    for csv_path in tqdm(csv_files, desc="Motions", unit="file"):
        motion = _load_csv(str(csv_path))
        end = args.end if args.end >= 0 else motion.shape[0]
        motion = motion[args.start : min(end, motion.shape[0]) : max(1, args.stride)]

        qpos = _build_qpos_sequence(model, motion, CSV_JOINT_ORDER)
        qvel = _compute_qvel(model, qpos, args.fps)

        body_pos_w = np.zeros((motion.shape[0], len(body_ids), 3))
        body_quat_w = np.zeros((motion.shape[0], len(body_ids), 4))
        body_lin_vel_w = np.zeros((motion.shape[0], len(body_ids), 3))
        body_ang_vel_w = np.zeros((motion.shape[0], len(body_ids), 3))
        joint_pos = np.zeros((motion.shape[0], len(joint_names)))
        joint_vel = np.zeros((motion.shape[0], len(joint_names)))

        for t in tqdm(
            range(motion.shape[0]),
            desc=f"Exporting {csv_path.stem}",
            unit="frame",
            leave=False,
        ):
            data.qpos[:] = qpos[t]
            data.qvel[:] = qvel[t]
            mujoco.mj_forward(model, data)

            body_pos_w[t] = data.xpos[body_ids]
            body_quat_w[t] = data.xquat[body_ids]
            body_ang_vel_w[t] = data.cvel[body_ids][:, 0:3]
            body_lin_vel_w[t] = data.cvel[body_ids][:, 3:6]

            joint_pos[t] = data.qpos[joint_qpos_addrs]
            joint_vel[t] = data.qvel[joint_dof_addrs]

        motion_dir = out_dir / csv_path.stem
        motion_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            motion_dir / "motion.npz",
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            joint_pos=joint_pos,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_vel=joint_vel,
        )

        meta = {"body_names": body_names, "joint_names": joint_names, "fps": args.fps, "length": motion.shape[0]}
        (motion_dir / "meta.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()

import numpy as np
import torch
import sys
import os

sys.path.insert(0, "C:/sportsai-backend/models/AthletePose3D")

YOLO_MODEL_SIZE = "yolo11x-pose.pt"
MOTIONAGFORMER_CKPT = "C:/sportsai-backend/models/AthletePose3D/model_params/motionagformer-s-ap3d.pth.tr"
CONFIDENCE_THRESHOLD = 0.7

def _calc_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def _extract_biomechanics(pose_seq: np.ndarray, technique: str) -> dict:
    biomechanics = {}

    if technique == "spike":
        angles = [_calc_angle(pose_seq[t, 6], pose_seq[t, 8], pose_seq[t, 10]) for t in range(len(pose_seq))]
        biomechanics["arm_cock_angle"] = float(np.max(angles))
        ankle_y = pose_seq[:, 15, 1]
        biomechanics["jump_height"] = float((ankle_y.max() - ankle_y.min()) * 100)
        hip_pos = pose_seq[:, 11, :2]
        dists = np.linalg.norm(np.diff(hip_pos, axis=0), axis=1)
        biomechanics["approach_speed"] = float(np.mean(dists) * 30)
        hip_center = (pose_seq[:, 11] + pose_seq[:, 12]) / 2
        wrist = pose_seq[:, 10]
        biomechanics["contact_point"] = float(np.linalg.norm(wrist - hip_center, axis=1).min())
        biomechanics["follow_through"] = float(np.std(angles) * 2)

    elif technique == "serve":
        biomechanics["shoulder_rotation"] = float(_calc_angle(pose_seq[len(pose_seq)//2, 5], pose_seq[len(pose_seq)//2, 11], pose_seq[len(pose_seq)//2, 6]))
        wrist_y = pose_seq[:, 10, 1]
        biomechanics["toss_height"] = float((wrist_y.max() - wrist_y.min()) * 100)
        trunk_angles = [_calc_angle(pose_seq[t, 0], pose_seq[t, 11], pose_seq[t, 15]) for t in range(len(pose_seq))]
        biomechanics["body_lean"] = float(np.mean(trunk_angles))
        biomechanics["step_timing"] = float(len(pose_seq) / 30.0)
        biomechanics["wrist_snap"] = float(np.std(pose_seq[:, 10, 0]) * 100)

    elif technique == "block":
        hand_y = pose_seq[:, 10, 1]
        biomechanics["hand_position"] = float(hand_y.max() * 100)
        shoulder_dist = np.linalg.norm(pose_seq[:, 5, :2] - pose_seq[:, 6, :2], axis=1)
        biomechanics["shoulder_width"] = float(np.mean(shoulder_dist) * 100)
        biomechanics["reaction_time"] = float(len(pose_seq) / 30.0 * 0.3)
        biomechanics["penultimate_step"] = float(np.linalg.norm(pose_seq[-2, 15] - pose_seq[-3, 15]) * 100)
        ankle_y_end = pose_seq[-5:, 15, 1]
        biomechanics["landing_balance"] = float(1.0 - np.std(ankle_y_end))

    elif technique == "dig":
        knee_angle = [_calc_angle(pose_seq[t, 11], pose_seq[t, 13], pose_seq[t, 15]) for t in range(len(pose_seq))]
        biomechanics["knee_bend"] = float(np.min(knee_angle))
        elbow_angle = [_calc_angle(pose_seq[t, 5], pose_seq[t, 7], pose_seq[t, 9]) for t in range(len(pose_seq))]
        biomechanics["arm_extension"] = float(np.mean(elbow_angle))
        hip_y = pose_seq[:, 11, 1]
        biomechanics["hip_drop"] = float((hip_y.max() - hip_y.min()) * 100)
        wrist_dist = np.linalg.norm(pose_seq[:, 9, :2] - pose_seq[:, 10, :2], axis=1)
        biomechanics["platform_angle"] = float(np.mean(wrist_dist) * 100)
        biomechanics["recovery_position"] = float(len(pose_seq) / 30.0)

    return biomechanics

def extract_pose(video_path: str, technique: str) -> dict:
    from ultralytics import YOLO
    import cv2

    yolo = YOLO(YOLO_MODEL_SIZE)
    cap = cv2.VideoCapture(video_path)
    pose_frames = []
    confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame, verbose=False)
        if not results or results[0].keypoints is None:
            continue
        kps = results[0].keypoints
        if kps.conf is None or len(kps.conf) == 0:
            continue
        best_idx = int(kps.conf.mean(dim=1).argmax())
        conf = float(kps.conf[best_idx].mean())
        if conf < CONFIDENCE_THRESHOLD:
            continue
        xy = kps.xy[best_idx].cpu().numpy()
        z = np.zeros((17, 1))
        pose_3d = np.concatenate([xy, z], axis=1)
        pose_frames.append(pose_3d)
        confidences.append(conf)

    cap.release()

    if len(pose_frames) < 5:
        raise ValueError(f"Too few valid frames ({len(pose_frames)}) in {video_path}")

    pose_seq = np.array(pose_frames)
    avg_conf = float(np.mean(confidences))
    biomechanics = _extract_biomechanics(pose_seq, technique)

    return {
        "pose_sequence_3d": pose_seq,
        "biomechanics": biomechanics,
        "average_confidence": round(avg_conf, 4)
    }



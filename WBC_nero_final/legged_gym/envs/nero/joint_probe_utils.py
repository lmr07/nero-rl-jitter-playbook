"""Isaac-independent definitions shared by training and offline hardware plans."""
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from nero_joint_contract import VELOCITY_LIMITS


MOTOR_GROUPS = ((1, 2), (3, 4), (5,), (6,), (7,))
TORQUE_LIMITS = (24., 24., 16., 16., 8., 8., 8.)


def find_checkpoint(root, run=-1, iteration=-1):
    """Numeric checkpoints only; exports cannot masquerade as training models."""
    if run in (-1, "-1", None):
        runs = [os.path.join(root, name) for name in os.listdir(root)
                if os.path.isfile(os.path.join(root, name, "joint_config.json"))]
        runs = [path for path in runs if any(re.fullmatch(r"model_\d+\.pt", n) for n in os.listdir(path))]
        if not runs:
            raise ValueError("No joint checkpoint under " + root)
        folder = max(runs, key=lambda path: os.path.getmtime(os.path.join(path, "joint_config.json")))
    else:
        folder = os.path.join(root, run)
    candidates = [(int(m.group(1)), name) for name in os.listdir(folder)
                  for m in [re.fullmatch(r"model_(\d+)\.pt", name)] if m]
    if iteration in (-1, None):
        if not candidates:
            raise ValueError("No numeric checkpoint in " + folder)
        iteration = max(candidates)[0]
    path = os.path.join(folder, "model_%d.pt" % iteration)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def joint_limits(urdf):
    root = ET.parse(urdf).getroot()
    limits = [root.find("joint[@name='nero_joint%d']/limit" % j) for j in range(1, 8)]
    lo = np.array([float(x.get("lower")) for x in limits])
    hi = np.array([float(x.get("upper")) for x in limits])
    return lo, hi, .7 * np.minimum(-lo, hi)


def identification(source):
    with open(source, "rb") as f:
        raw = f.read()
    results = sorted(json.loads(raw)["results"], key=lambda r: r["joint"])
    if [r["joint"] for r in results] != list(range(1, 8)):
        raise ValueError("Identification must contain exactly J1..J7")
    delays = [float(r["delay_ms"]) for r in results]
    if not all(np.isfinite(delays)) or min(delays) < 0:
        raise ValueError("Invalid identification delays")
    return results, {
        "source": os.path.abspath(source), "sha256": hashlib.sha256(raw).hexdigest(),
        "joint_delays_ms": delays, "mean_delay_s": float(np.mean(delays) / 1000),
        "interpretation": "lumped identified lag, applied once to position commands; not a separately measured CAN delay",
    }


def fixed_pose_urdf(source, destination, active_joint, pose):
    """R_origin * R_axis(q) preserves the locked joint's true transform/inertia."""
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", os.path.abspath(os.path.join(os.path.dirname(source), mesh.get("filename"))))
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name == "nero_joint%d" % active_joint or joint.get("type") == "fixed":
            continue
        angle = float(pose[int(name[len("nero_joint"):]) - 1]) if name.startswith("nero_joint") else 0.
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        if joint.get("type") in ("revolute", "continuous"):
            r = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            r = r * Rotation.from_rotvec(axis * angle)
            origin.set("rpy", " ".join("%.15g" % v for v in r.as_euler("xyz")))
        elif angle != 0:
            raise ValueError("Nonzero prismatic lock not supported")
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "mimic"):
            for node in joint.findall(tag):
                joint.remove(node)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def pd_candidates(source, scale_factors=(.8, 1., 1.25), training_kp=48., training_kd=1.5):
    """Same Cartesian alpha/beta grid for every joint in a motor group.

    These are offline candidates, NOT approved gains or an automatic hardware sweep.
    """
    rows, provenance = identification(source)
    groups = []
    for joints in MOTOR_GROUPS:
        alpha = float(np.mean([rows[j-1]["alpha_sdk"] for j in joints]))
        beta = float(np.mean([rows[j-1]["beta_sdk"] for j in joints]))
        candidates = []
        for a in scale_factors:
            for b in scale_factors:
                candidates.append(dict(alpha=alpha*a, beta=beta*b,
                                       kp=training_kp/(alpha*a), kd=training_kd/(beta*b)))
        groups.append(dict(joints=joints, candidates=candidates))
    return dict(hardware_approved=False, identification=provenance, groups=groups,
                training_pd=dict(kp=training_kp, kd=training_kd),
                selection="Use one common candidate per motor group; score all its joints and held-out poses. Stability alone does not identify unique alpha/beta.")

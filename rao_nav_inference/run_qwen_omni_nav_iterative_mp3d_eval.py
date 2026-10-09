#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import habitat_sim
import networkx as nx
import numpy as np
import requests
import torch
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector

from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from rao_nav_inference.configuration import get_config

from rao_nav_inference import qwen_omni_belief_fusion_infer as fusion
from rao_nav_inference.localization_expert.inference import LoadedLocalizationExpert
from rao_nav_inference.paths import (
    BINAURAL_RIRS_DIR,
    DEFAULT_RESULTS_DIR,
    LOCALIZATION_CHECKPOINT,
    MP3D_SCENES_DIR,
    SAVI_BELIEF_CHECKPOINT,
    SAVI_CONFIG,
    TEST_CONTENT_DIR,
    TEST_SOUNDS_DIR,
)




ACTION_MAP = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
ACTION_INV = {v: k for k, v in ACTION_MAP.items()}




TURN_ANGLE_DEG = 90.0
TURN_ANGLE_RAD = np.radians(TURN_ANGLE_DEG)
FORWARD_STEP_M = 1.0
WAYPOINT_REACH_THRESH_M = 0.36
MAX_ITERATIONS_PER_EPISODE = 20
MAX_EPISODE_STEPS = 500
OFFICIAL_MAX_EPISODE_STEPS = 200
OFFICIAL_RGB_RESOLUTION = 256
_TOPDOWN_MASK_CACHE: Dict[Tuple[Any, ...], np.ndarray] = {}

PATH_COLORS_BGR = [
    (0, 0, 255),    
    (0, 200, 0),    
    (255, 100, 0),  
    (0, 220, 220),  
]





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen-Omni iterative navigation eval on MP3D test set.")
    p.add_argument("--exp-config", type=str, default=str(SAVI_CONFIG))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--dataset-content-dir", type=str, default=str(TEST_CONTENT_DIR))
    p.add_argument("--num-episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=OFFICIAL_MAX_EPISODE_STEPS)
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS_PER_EPISODE)
    p.add_argument("--omni-url", type=str, default="http://127.0.0.1:6006/v1/omni/inference")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--belief-ckpt", type=str, default=str(SAVI_BELIEF_CHECKPOINT))
    p.add_argument(
        "--localization-ckpt",
        type=str,
        default=str(LOCALIZATION_CHECKPOINT),
        help="Qwen-conditioned Localization Expert checkpoint used for waypoint prediction.",
    )
    p.add_argument(
        "--localization-device",
        type=str,
        default="cpu",
        help="Torch device for the Localization Expert. CPU avoids competing with the Qwen server GPU.",
    )
    p.add_argument("--fusion-alpha", type=float, default=1.0)
    p.add_argument("--fusion-beta", type=float, default=0.8)
    p.add_argument("--belief-weighting-factor", type=float, default=0.5)
    p.add_argument("--meters-per-pixel", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="Resume from per-episode stats already saved in --save-dir.")
    p.add_argument("--save-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    return p.parse_args()





def _count_episodes(content_dir: Path) -> int:
    if not content_dir.is_dir():
        raise FileNotFoundError(f"Dataset content dir not found: {content_dir}")
    total = 0
    for p in sorted(content_dir.glob("*.json.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        eps = data["episodes"] if isinstance(data, dict) else data
        total += len(eps)
    return total





def _build_eval_config(args: argparse.Namespace, episode_cap: int):
    opts = [
        "EVAL.SPLIT", args.split,
        "USE_SYNC_VECENV", "True",
        "NUM_PROCESSES", "1",
        "RL.DDPPO.pretrained", "False",
        "TEST_EPISODE_COUNT", str(max(int(episode_cap), 1)),
    ]
    cfg = get_config(args.exp_config, opts, model_dir="data/models/output",
                     run_type="eval", overwrite=False)
    cfg.defrost()
    cfg.NUM_PROCESSES = 1
    cfg.USE_SYNC_VECENV = True
    cfg.EVAL.SPLIT = args.split
    sensors = list(cfg.TASK_CONFIG.TASK.SENSORS)
    if "POSE_SENSOR" not in sensors:
        sensors.append("POSE_SENSOR")
    cfg.TASK_CONFIG.TASK.SENSORS = sensors
    content_dir = Path(args.dataset_content_dir).expanduser().resolve()
    split_json = content_dir.parent / f"{args.split}.json.gz"
    cfg.TASK_CONFIG.defrost()
    cfg.TASK_CONFIG.DATASET.DATA_PATH = str(split_json)
    cfg.TASK_CONFIG.DATASET.SPLIT = args.split
    cfg.TASK_CONFIG.DATASET.CONTENT_SCENES = ["*"]
    cfg.TASK_CONFIG.DATASET.SCENES_DIR = str(MP3D_SCENES_DIR)
    cfg.TASK_CONFIG.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = str(BINAURAL_RIRS_DIR)
    cfg.TASK_CONFIG.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = str(TEST_SOUNDS_DIR.parent)
    cfg.SEED = int(args.seed)
    cfg.TASK_CONFIG.SEED = int(args.seed)
    
    
    
    
    total_available = _count_episodes(content_dir)
    cfg.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = (
        int(episode_cap) if int(episode_cap) < total_available else -1
    )
    cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = OFFICIAL_RGB_RESOLUTION
    cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = OFFICIAL_RGB_RESOLUTION
    cfg.TASK_CONFIG.freeze()
    cfg.freeze()
    return cfg





def _odom_to_world(odom_xy: np.ndarray, start_pos: np.ndarray,
                   start_rot: List[float]) -> np.ndarray:
    local = np.array([float(odom_xy[1]), 0.0, -float(odom_xy[0])], dtype=np.float32)
    rot = quaternion_from_coeff(start_rot)
    return (np.asarray(start_pos, dtype=np.float32)
            + quaternion_rotate_vector(rot, local).astype(np.float32))


def _odom_to_base(odom_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    angle = -float(pose[2])
    delta = odom_xy - pose[:2]
    dt = float(np.arctan2(delta[1], delta[0])) - angle
    d = float(np.linalg.norm(delta))
    return np.array([d * np.cos(dt), d * np.sin(dt)], dtype=np.float32)


def _base_to_odom(base_xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    angle = -float(pose[2])
    d = float(np.linalg.norm(base_xy))
    theta = float(np.arctan2(base_xy[1], base_xy[0]))
    return np.array([pose[0] + d * np.cos(theta + angle),
                     pose[1] + d * np.sin(theta + angle)], dtype=np.float32)


def _get_agent_world(obs: Dict, ep) -> np.ndarray:
    pose = np.asarray(obs.get("pose", np.zeros(3)), dtype=np.float32)
    return _odom_to_world(pose[:2], np.asarray(ep.start_position, dtype=np.float32),
                          list(ep.start_rotation))





def _write_temp_wav(audiogoal: np.ndarray, sr: int, tag: str = "iter") -> str:
    pcm = np.clip(np.asarray(audiogoal, dtype=np.float32).T, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    tmp = Path(tempfile.gettempdir()) / f"qwen_nav_{tag}_audio.wav"
    with wave.open(tmp.as_posix(), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
    return tmp.as_posix()


def _write_temp_rgb(rgb: np.ndarray, tag: str = "iter") -> str:
    img = np.asarray(rgb)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    tmp = Path(tempfile.gettempdir()) / f"qwen_nav_{tag}_rgb.png"
    cv2.imwrite(tmp.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return tmp.as_posix()





class NavSimCache:
    def __init__(self):
        self._sim: Optional[habitat_sim.Simulator] = None
        self._scene_id: str = ""

    @property
    def pathfinder(self):
        assert self._sim is not None
        return self._sim.pathfinder

    def ensure(self, scene_id: str, scene_dataset_cfg: str):
        if self._scene_id == scene_id and self._sim is not None:
            return
        self.close()
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = scene_id
        backend.scene_dataset_config_file = scene_dataset_cfg
        backend.enable_physics = False
        backend.create_renderer = False
        backend.requires_textures = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
        navmesh = Path(scene_id).with_suffix(".navmesh")
        if not navmesh.exists():
            raise FileNotFoundError(f"Navmesh not found: {navmesh}")
        if not self._sim.pathfinder.load_nav_mesh(navmesh.as_posix()):
            raise RuntimeError(f"Failed to load navmesh: {navmesh}")
        self._scene_id = scene_id

    def close(self):
        if self._sim is not None:
            self._sim.close()
            self._sim = None
            self._scene_id = ""





def build_localization_candidate_priors(qwen_top3: List[str]) -> List[np.ndarray]:
    mixed = fusion.build_qwen_prior(qwen_top3, fusion.TASK_CATEGORIES_21)
    ranked_priors = []
    for category in qwen_top3[:3]:
        prior = np.full(len(fusion.TASK_CATEGORIES_21), 1e-6, dtype=np.float32)
        if category in fusion.TASK_CATEGORIES_21:
            prior[fusion.TASK_CATEGORIES_21.index(category)] = 1.0
        prior /= prior.sum()
        ranked_priors.append(prior)
    while len(ranked_priors) < 3:
        ranked_priors.append(mixed.copy())
    return [ranked_priors[1], ranked_priors[2], mixed, ranked_priors[0]]


def predict_waypoint_candidates(obs: Dict, ep, label_model, localization_expert,
                                qwen_top3: List[str], args) -> Tuple[List[np.ndarray], str, List[np.ndarray]]:
    
    spec = np.asarray(obs["spectrogram"], dtype=np.float32)
    spec_t = torch.from_numpy(spec).unsqueeze(0).permute(0, 3, 1, 2)

    with torch.no_grad():
        cat_logits = label_model(spec_t).squeeze(0)
        cat_prob = torch.softmax(cat_logits, dim=0).cpu().numpy()

    qwen_prior = fusion.build_qwen_prior(qwen_top3, fusion.TASK_CATEGORIES_21)
    eps = 1e-8
    fused_log = (float(args.fusion_alpha) * np.log(cat_prob + eps)
                 + float(args.fusion_beta) * np.log(qwen_prior + eps))
    fused_prob = np.exp(fused_log - np.max(fused_log))
    fused_prob /= fused_prob.sum()
    fused_label = fusion.TASK_CATEGORIES_21[int(np.argmax(fused_prob))]

    candidate_priors = build_localization_candidate_priors(qwen_top3)
    pg_bases = [
        np.asarray(localization_expert.predict_pointgoal(spec, prior), dtype=np.float32)
        for prior in candidate_priors
    ]
    if np.sum(spec) == 0.0:
        pg_bases = [np.array([10.0, 10.0], dtype=np.float32) for _ in range(4)]

    pose = np.asarray(obs.get("pose", np.zeros(3)), dtype=np.float32)
    target_worlds = [
        _odom_to_world(
            _base_to_odom(pg_base, pose),
            np.asarray(ep.start_position, dtype=np.float32),
            list(ep.start_rotation),
        )
        for pg_base in pg_bases
    ]
    return target_worlds, fused_label, pg_bases





def generate_4_paths(sound_sim, start_w: np.ndarray,
                     goal_candidates_w: List[np.ndarray]) -> List[List[np.ndarray]]:
    
    graph = sound_sim.graph
    orientation_lattice = int(sound_sim.get_orientation()) % 90
    executable_edges = []
    for a, b in graph.edges():
        p1 = np.asarray(graph.nodes[a]["point"], dtype=np.float32)
        p2 = np.asarray(graph.nodes[b]["point"], dtype=np.float32)
        direction = int(np.around(np.rad2deg(np.arctan2(
            float(p2[2] - p1[2]), float(p2[0] - p1[0])
        )))) % 360
        if (direction - orientation_lattice) % 90 == 0:
            executable_edges.append((a, b))
    executable_graph = graph.edge_subgraph(executable_edges)
    start_node = sound_sim._receiver_position_index
    if start_node not in executable_graph:
        reachable_nodes = [start_node]
    else:
        reachable_nodes = list(nx.node_connected_component(executable_graph, start_node))
    node_ids = reachable_nodes
    node_points = np.asarray(
        [graph.nodes[n]["point"] for n in node_ids], dtype=np.float32
    )

    start_w = np.asarray(graph.nodes[start_node]["point"], dtype=np.float32)

    def _node_path(source, target):
        try:
            nodes = nx.shortest_path(executable_graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return [np.asarray(graph.nodes[n]["point"], dtype=np.float32) for n in nodes]

    goals = [np.asarray(x, dtype=np.float32) for x in goal_candidates_w[:4]]
    while len(goals) < 4:
        goals.append(goals[-1].copy() if goals else start_w.copy())
    paths: List[List[np.ndarray]] = []
    used_nodes = set()
    for goal in goals:
        order = np.argsort(np.linalg.norm(node_points - goal[None, :], axis=1))
        goal_node = next((node_ids[int(i)] for i in order if node_ids[int(i)] not in used_nodes), node_ids[int(order[0])])
        used_nodes.add(goal_node)
        goal_snap = np.asarray(graph.nodes[goal_node]["point"], dtype=np.float32)
        path = _node_path(start_node, goal_node)
        if not path:
            path = [start_w.copy(), goal_snap.copy()]
        paths.append(path)
    return paths





def _draw_star(img, rc, radius, color_bgr):
    cx, cy = rc[1], rc[0]
    pts = []
    for i in range(10):
        a = -np.pi / 2.0 + i * (np.pi / 5.0)
        r = float(radius) if i % 2 == 0 else float(radius) * 0.45
        pts.append([int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color_bgr)


def render_topdown_4paths(pf, paths: List[List[np.ndarray]],
                          start_w: np.ndarray, goal_candidates_w: List[np.ndarray],
                          mpp: float = 0.02, tag: str = "iter") -> str:
    
    all_pts = [start_w] + [np.asarray(x, dtype=np.float32) for x in goal_candidates_w]
    for pa in paths:
        all_pts.extend(pa)
    ground_y = float(np.median([float(p[1]) for p in all_pts]))

    bounds = pf.get_bounds()
    bounds_key = tuple(round(float(v), 2) for p in bounds for v in p)
    cache_key = (id(pf), bounds_key, round(float(mpp), 5), round(float(ground_y), 2))
    nav_mask = _TOPDOWN_MASK_CACHE.get(cache_key)
    if nav_mask is None:
        nav_mask = pf.get_topdown_view(mpp, ground_y)
        if nav_mask is None or nav_mask.size == 0:
            nav_mask = np.ones((200, 200), dtype=bool)
        nav_mask = np.asarray(nav_mask, dtype=bool)
        _TOPDOWN_MASK_CACHE[cache_key] = nav_mask
    h, w = nav_mask.shape
    img = np.full((h, w, 3), 55, dtype=np.uint8)
    img[nav_mask] = 230

    sx = float(min(bounds[0][0], bounds[1][0]))
    sz = float(min(bounds[0][2], bounds[1][2]))

    def to_rc(p):
        c = int(np.clip((float(p[0]) - sx) / mpp, 0, w - 1))
        r = int(np.clip((float(p[2]) - sz) / mpp, 0, h - 1))
        return r, c

    for i, pa in enumerate(paths):
        color = PATH_COLORS_BGR[i % len(PATH_COLORS_BGR)]
        rcs = [to_rc(p) for p in pa]
        for j in range(len(rcs) - 1):
            cv2.line(img, (rcs[j][1], rcs[j][0]), (rcs[j + 1][1], rcs[j + 1][0]),
                     color, 4 if i == 0 else 3, cv2.LINE_AA)
        for rc in rcs:
            cv2.circle(img, (rc[1], rc[0]), 4, color, -1, cv2.LINE_AA)

    rc_s = to_rc(start_w)
    cv2.circle(img, (rc_s[1], rc_s[0]), 12, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (rc_s[1], rc_s[0]), 14, (0, 0, 0), 2, cv2.LINE_AA)

    for i, pa in enumerate(paths):
        if pa:
            _draw_star(img, to_rc(pa[-1]), 12, PATH_COLORS_BGR[i % len(PATH_COLORS_BGR)])

    y0 = 30
    for i in range(min(len(paths), 4)):
        y = y0 + i * 28
        cv2.circle(img, (20, y), 7, PATH_COLORS_BGR[i], -1, cv2.LINE_AA)
        cv2.putText(img, f"Candidate {i + 1}", (35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
    y = y0 + 4 * 28
    cv2.circle(img, (20, y), 7, (255, 120, 0), -1, cv2.LINE_AA)
    cv2.putText(img, "Start", (35, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
    y += 28
    _draw_star(img, (y, 20), 7, (0, 0, 255))
    cv2.putText(img, "Candidate target", (35, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)

    
    
    
    
    max_side = max(img.shape[:2])
    if max_side > 1024:
        scale = 1024.0 / float(max_side)
        img = cv2.resize(
            img,
            (max(1, int(round(img.shape[1] * scale))),
             max(1, int(round(img.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    out = Path(tempfile.gettempdir()) / f"qwen_nav_topdown_{tag}.png"
    cv2.imwrite(out.as_posix(), img)
    return out.as_posix()


def _compose_images(topdown_path: str, rgb_path: str,
                    tag: str = "iter") -> str:
    td = cv2.imread(topdown_path)
    rgb = cv2.imread(rgb_path)
    if td is None or rgb is None:
        return topdown_path
    th = max(td.shape[0], rgb.shape[0], 256)
    td2 = cv2.resize(td, (int(td.shape[1] * th / max(1, td.shape[0])), th))
    rgb2 = cv2.resize(rgb, (int(rgb.shape[1] * th / max(1, rgb.shape[0])), th))
    comp = np.hstack([td2, rgb2])
    out = Path(tempfile.gettempdir()) / f"qwen_nav_composite_{tag}.png"
    cv2.imwrite(out.as_posix(), comp)
    return out.as_posix()





def _call_qwen_omni(omni_url: str, timeout: float, **kw) -> str:
    payload = {
        "temperature": kw.get("temperature", 0.1),
        "question": "",
        "user_prompt": kw.get("user_prompt", ""),
        "system_prompt": kw.get("system_prompt", ""),
        "video_path": kw.get("video_path", ""),
        "audio_path": kw.get("audio_path", None),
        "multi_video_path": kw.get("multi_video_path", "None"),
        "history_actions": kw.get("history_actions", ["None"]),
        "collision": kw.get("collision", False),
        "max_new_tokens": kw.get("max_new_tokens", 64),
        "lnr_goal": kw.get("lnr_goal", None),
        "history_image_paths": kw.get("history_image_paths", []),
    }
    try:
        resp = requests.post(omni_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip()
    except Exception as e:
        print(f"[WARN] Qwen-Omni call failed: {e}")
        return ""


def ask_choose_path_or_stop(omni_url: str, timeout: float,
                            composite_path: str, audio_path: str,
                            fused_label: str,
                            history_actions: List[str],
                            history_image_paths: Optional[List[str]] = None,
                            temperature: float = 0.1,
                            max_new_tokens: int = 64) -> Tuple[Optional[int], bool, str]:
    
    sys_prompt = (
        "You are an audio-visual navigation agent. "
        "You see a composite image: LEFT is a topdown map with 4 candidate target positions and their paths, "
        "RIGHT is your current camera view. "
        "Choose the best path to reach the sound source, or output STOP if you "
        "believe you have already arrived at the sound source."
    )
    user_prompt = (
        "The topdown map shows 4 candidate target positions and shortest paths:\n"
        "- Path 1 (Red)\n"
        "- Path 2 (Green)\n"
        "- Path 3 (Blue)\n"
        "- Path 4 (Yellow)\n"
        "Blue circle = current position. Each colored star is that path's candidate target.\n"
        f"Predicted sound category: {fused_label}.\n"
        f"Recent actions: {history_actions[-5:] if history_actions else ['None']}.\n"
        "If you believe you have reached the sound source, output exactly: STOP\n"
        "Otherwise, output exactly one number (1, 2, 3, or 4) for your chosen path."
    )
    raw = _call_qwen_omni(
        omni_url, timeout,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        video_path=composite_path, audio_path=audio_path,
        history_actions=history_actions[-5:] if history_actions else ["None"],
        lnr_goal=fused_label,
        history_image_paths=history_image_paths or [],
        temperature=temperature, max_new_tokens=max_new_tokens,
    )
    if "STOP" in raw.upper():
        return None, True, raw
    m = re.search(r"[1-4]", raw)
    idx = (int(m.group()) - 1) if m else 0
    return idx, False, raw





def _rotate_xz(d: np.ndarray, angle_rad: float) -> np.ndarray:
    
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([d[0] * c + d[1] * s,
                     -d[0] * s + d[1] * c], dtype=np.float32)


def _initial_heading_xz(start_rot: List[float]) -> np.ndarray:
    
    fwd = quaternion_rotate_vector(
        quaternion_from_coeff(start_rot),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    ).astype(np.float32)
    d = np.array([fwd[0], fwd[2]], dtype=np.float32)
    n = max(float(np.linalg.norm(d)), 1e-8)
    return d / n


def precompute_path_actions(
    waypoints_world: List[np.ndarray],
    heading_xz: np.ndarray,
) -> Tuple[List[int], np.ndarray]:
    
    actions: List[int] = []
    cur_dir = heading_xz.copy()

    for i in range(len(waypoints_world) - 1):
        seg = waypoints_world[i + 1] - waypoints_world[i]
        seg_xz = np.array([seg[0], seg[2]], dtype=np.float32)
        dist = float(np.linalg.norm(seg_xz))
        if dist < 0.05:
            continue
        seg_dir = seg_xz / dist

        cross_y = cur_dir[0] * seg_dir[1] - cur_dir[1] * seg_dir[0]
        dot = float(np.dot(cur_dir, seg_dir))
        angle = float(np.arctan2(cross_y, dot))

        n_turns = int(round(abs(angle) / TURN_ANGLE_RAD))
        if angle > 0:
            turn_act = ACTION_MAP["TURN_LEFT"]
            actual_angle = n_turns * TURN_ANGLE_RAD
        else:
            turn_act = ACTION_MAP["TURN_RIGHT"]
            actual_angle = -n_turns * TURN_ANGLE_RAD
        actions.extend([turn_act] * n_turns)
        cur_dir = _rotate_xz(cur_dir, actual_angle)

        n_fwd = max(1, int(round(dist / FORWARD_STEP_M)))
        actions.extend([ACTION_MAP["MOVE_FORWARD"]] * n_fwd)

    return actions, cur_dir


def precompute_soundspaces_actions(
    waypoints_world: List[np.ndarray], orientation_deg: int
) -> List[int]:
    
    actions: List[int] = []
    orientation = int(orientation_deg) % 360
    for p1, p2 in zip(waypoints_world[:-1], waypoints_world[1:]):
        direction = int(np.around(np.rad2deg(np.arctan2(
            float(p2[2] - p1[2]), float(p2[0] - p1[0])
        )))) % 360
        delta = (direction - orientation) % 360
        if delta == 270:
            orientation = (orientation - 90) % 360
            actions.append(ACTION_MAP["TURN_LEFT"])
        elif delta == 90:
            orientation = (orientation + 90) % 360
            actions.append(ACTION_MAP["TURN_RIGHT"])
        elif delta == 180:
            orientation = (orientation - 180) % 360
            actions.extend([ACTION_MAP["TURN_RIGHT"], ACTION_MAP["TURN_RIGHT"]])
        elif delta != 0:
            raise RuntimeError(f"Non-cardinal SoundSpaces graph edge: delta={delta}")
        actions.append(ACTION_MAP["MOVE_FORWARD"])
    return actions





def _point_to_polyline_xz(point: np.ndarray, waypoints: List[np.ndarray]) -> float:
    
    p = np.asarray(point, dtype=np.float32)[[0, 2]]
    if not waypoints:
        return float("nan")
    pts = [np.asarray(x, dtype=np.float32)[[0, 2]] for x in waypoints]
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def follow_path(envs, obs: Dict, ep, waypoints_world: List[np.ndarray],
                heading_xz: np.ndarray,
                max_budget: int,
                action_seq: Optional[List[int]] = None,
                ) -> Tuple[Dict, int, bool, Dict, List[str], np.ndarray, Dict[str, Any]]:
    
    if action_seq is None:
        action_seq, _ = precompute_path_actions(waypoints_world, heading_xz)
    if len(action_seq) > max_budget:
        action_seq = action_seq[:max_budget]

    info: Dict[str, Any] = {}
    actions_taken: List[str] = []
    total = 0
    executed_heading = heading_xz.copy()
    measured_positions = [_get_agent_world(obs, ep)]
    forward_positions: List[np.ndarray] = []

    for act in action_seq:
        outputs = envs.step([act])
        obs_l, _, dones, infos = [list(x) for x in zip(*outputs)]
        obs, info = obs_l[0], infos[0]
        actions_taken.append(ACTION_INV.get(act, "MOVE_FORWARD"))
        total += 1
        measured = _get_agent_world(obs, ep)
        measured_positions.append(measured)
        if act == ACTION_MAP["MOVE_FORWARD"]:
            forward_positions.append(measured)
        elif act == ACTION_MAP["TURN_LEFT"]:
            executed_heading = _rotate_xz(executed_heading, TURN_ANGLE_RAD)
        elif act == ACTION_MAP["TURN_RIGHT"]:
            executed_heading = _rotate_xz(executed_heading, -TURN_ANGLE_RAD)

        if bool(dones[0]):
            break

    planned = [np.asarray(p, dtype=np.float32) for p in waypoints_world]
    start = measured_positions[0]
    end = measured_positions[-1]
    route_end = planned[-1] if planned else start
    samples = forward_positions if forward_positions else measured_positions
    cross_track = [_point_to_polyline_xz(p, planned) for p in samples]
    planned_length = sum(
        float(np.linalg.norm((b - a)[[0, 2]])) for a, b in zip(planned[:-1], planned[1:])
    )
    actual_length = sum(
        float(np.linalg.norm((b - a)[[0, 2]]))
        for a, b in zip(measured_positions[:-1], measured_positions[1:])
    )
    effective_forward = sum(
        float(np.linalg.norm((b - a)[[0, 2]])) > 0.05
        for a, b, name in zip(measured_positions[:-1], measured_positions[1:], actions_taken)
        if name == "MOVE_FORWARD"
    )
    diagnostics: Dict[str, Any] = {
        "planned_waypoints": len(planned),
        "planned_waypoints_world": [p.tolist() for p in planned],
        "planned_length_m": planned_length,
        "planned_start_world": start.tolist() if not planned else planned[0].tolist(),
        "planned_end_world": route_end.tolist(),
        "actual_start_world": start.tolist(),
        "actual_end_world": end.tolist(),
        "actual_trajectory_world": [p.tolist() for p in measured_positions],
        "actual_path_length_m": actual_length,
        "actual_displacement_m": float(np.linalg.norm((end - start)[[0, 2]])),
        "endpoint_distance_before_m": float(np.linalg.norm((start - route_end)[[0, 2]])),
        "endpoint_distance_after_m": float(np.linalg.norm((end - route_end)[[0, 2]])),
        "endpoint_progress_m": float(
            np.linalg.norm((start - route_end)[[0, 2]])
            - np.linalg.norm((end - route_end)[[0, 2]])
        ),
        "mean_cross_track_error_m": float(np.mean(cross_track)) if cross_track else None,
        "max_cross_track_error_m": float(np.max(cross_track)) if cross_track else None,
        "commanded_forward_steps": actions_taken.count("MOVE_FORWARD"),
        "effective_forward_steps": int(effective_forward),
        "executed_actions": len(actions_taken),
        "executed_action_names": list(actions_taken),
        "episode_done_during_path": bool(total and bool(dones[0])),
    }

    return obs, total, bool(total and bool(dones[0])), info, actions_taken, executed_heading, diagnostics





def _extract_optimal_num_actions(ep) -> Optional[int]:
    paths = getattr(ep, "shortest_paths", None)
    if not isinstance(paths, list) or not paths:
        return None
    seq = paths[0]
    if not isinstance(seq, list) or not seq:
        return None
    cnt = 0
    for a in seq:
        if isinstance(a, (int, np.integer)):
            cnt += 1
        elif hasattr(a, "action"):
            cnt += 1
    return cnt if cnt > 0 else None


def _compute_sna(success: float, opt_actions: Optional[int],
                 num_steps: int) -> Optional[float]:
    if opt_actions is None:
        return None
    o = max(1, int(opt_actions))
    p = max(1, int(num_steps))
    return float(success) * (float(o) / float(max(o, p)))


def _extract_float_metric(info: Dict, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _extract_remaining_distance(info: Dict, fallback: float) -> float:
    for k in ["distance_to_goal", "dtg", "remaining_distance"]:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return fallback





def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stats_path = save_dir / f"{args.split}_nav_iterative_stats.json"
    summary_path = save_dir / f"{args.split}_nav_iterative_summary.json"

    content_dir = Path(args.dataset_content_dir).expanduser().resolve()
    num_target = int(args.num_episodes)
    if num_target < 0:
        num_target = _count_episodes(content_dir)
        print(f"[info] evaluating all {num_target} episodes in {content_dir}")

    cfg = _build_eval_config(args, episode_cap=num_target)
    envs = construct_envs(cfg, get_env_class(cfg.ENV_NAME), auto_reset_done=True)
    obs = envs.reset()[0]

    raw_ckpt = fusion.load_checkpoint_compat(args.belief_ckpt)
    belief_sd = fusion.extract_belief_state_dict(raw_ckpt)
    label_model = fusion.build_label_classifier()
    fusion.load_label_weights_from_belief(label_model, belief_sd)
    label_model.eval()

    localization_expert = LoadedLocalizationExpert(
        args.localization_ckpt, device=args.localization_device
    )

    sample_rate = int(cfg.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE)
    scene_dataset_cfg = str(cfg.TASK_CONFIG.SIMULATOR.SCENE_DATASET)
    nav_cache = NavSimCache()

    results: List[Dict[str, Any]] = []
    if args.resume and stats_path.is_file():
        loaded = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list) or len(loaded) > num_target:
            raise ValueError(f"Invalid resume stats: {stats_path}")
        results = loaded
        for index, previous in enumerate(results):
            current = envs.current_episodes()[0]
            current_key = (Path(str(current.scene_id)).stem, str(current.episode_id))
            previous_key = (str(previous.get("scene")), str(previous.get("episode_id")))
            if current_key != previous_key:
                raise RuntimeError(
                    f"Resume episode order mismatch at {index}: {current_key} != {previous_key}"
                )
            outputs = envs.step([ACTION_MAP["STOP"]])
            obs_l, _, _, _ = [list(x) for x in zip(*outputs)]
            obs = obs_l[0]
        print(f"[resume] loaded and skipped {len(results)} completed episodes from {stats_path}")

    while len(results) < num_target:
        ep = envs.current_episodes()[0]
        scene_id = str(ep.scene_id)
        scene_name = Path(scene_id).stem
        ep_id = str(getattr(ep, "episode_id", len(results)))
        sound_id = str(getattr(ep, "sound_id", ""))

        nav_cache.ensure(scene_id, scene_dataset_cfg)

        total_steps = 0
        done = False
        iteration = 0
        history_actions: List[str] = []
        history_rgb_paths: List[str] = []
        path_execution: List[Dict[str, Any]] = []
        final_info: Dict[str, Any] = {}
        last_pg_odoms: List[Optional[np.ndarray]] = [None, None, None, None]
        w_factor = float(args.belief_weighting_factor)
        is_silent_first = bool(np.sum(np.asarray(obs.get("spectrogram", np.zeros(1)), dtype=np.float32)) == 0.0)
        heading_xz = _initial_heading_xz(list(ep.start_rotation))

        
        audio_wav = _write_temp_wav(np.asarray(obs["audiogoal"]), sample_rate, tag="cls")
        placeholder_img = fusion._ensure_placeholder_image("")
        qwen_top3, qwen_raw_top3 = fusion.run_qwen_top3_online(
            omni_url=args.omni_url,
            timeout_sec=float(args.request_timeout),
            audio_file=audio_wav,
            candidate_classes=fusion.TASK_CATEGORIES_21,
            max_new_tokens=int(args.max_new_tokens),
            placeholder_image=placeholder_img,
        )

        while (not done
               and total_steps < int(args.max_steps)
               and iteration < int(args.max_iterations)):

            
            target_candidates_w, fused_label, pg_bases = predict_waypoint_candidates(
                obs, ep, label_model, localization_expert, qwen_top3, args)
            agent_w = _get_agent_world(obs, ep)

            
            pose = np.asarray(obs.get("pose", np.zeros(3)), dtype=np.float32)
            spec_nz = bool(np.sum(np.asarray(obs["spectrogram"], dtype=np.float32)) != 0.0)
            target_candidates_w = []
            for candidate_index, pg_base in enumerate(pg_bases):
                last_pg_odom = last_pg_odoms[candidate_index]
                if spec_nz:
                    if last_pg_odom is not None:
                        pg_base_smoothed = ((1.0 - w_factor) * pg_base
                                            + w_factor * _odom_to_base(last_pg_odom, pose))
                    else:
                        pg_base_smoothed = pg_base
                    pg_odom_now = _base_to_odom(pg_base_smoothed, pose)
                    last_pg_odoms[candidate_index] = pg_odom_now
                else:
                    pg_odom_now = last_pg_odom if last_pg_odom is not None else _base_to_odom(pg_base, pose)
                    last_pg_odoms[candidate_index] = pg_odom_now
                target_candidates_w.append(_odom_to_world(
                    pg_odom_now,
                    np.asarray(ep.start_position, dtype=np.float32),
                    list(ep.start_rotation),
                ))

            
            sound_sim = envs.workers[0]._env.habitat_env.sim
            paths_4 = generate_4_paths(sound_sim, agent_w, target_candidates_w)
            snapped_candidate_targets = [path[-1] for path in paths_4]

            
            td_path = render_topdown_4paths(
                nav_cache.pathfinder, paths_4, agent_w, snapped_candidate_targets,
                mpp=float(args.meters_per_pixel), tag=f"ep{len(results)}_it{iteration}")
            rgb_path = _write_temp_rgb(np.asarray(obs["rgb"]), tag=f"ep{len(results)}_it{iteration}")
            history_rgb_paths.append(rgb_path)
            audio_path = _write_temp_wav(np.asarray(obs["audiogoal"]), sample_rate,
                                         tag=f"ep{len(results)}_it{iteration}")
            composite = _compose_images(td_path, rgb_path, tag=f"ep{len(results)}_it{iteration}")

            
            chosen_idx, should_stop, choose_raw = ask_choose_path_or_stop(
                omni_url=args.omni_url, timeout=float(args.request_timeout),
                composite_path=composite, audio_path=audio_path,
                fused_label=fused_label,
                history_actions=history_actions,
                history_image_paths=history_rgb_paths,
                temperature=float(args.temperature),
                max_new_tokens=int(args.max_new_tokens),
            )
            if should_stop:
                outputs = envs.step([ACTION_MAP["STOP"]])
                obs_l, _, dones, infos = [list(x) for x in zip(*outputs)]
                obs = obs_l[0]
                done = True
                final_info = infos[0]
                history_actions.append("STOP")
                total_steps += 1
                break

            
            chosen_path = paths_4[chosen_idx] if chosen_idx is not None else paths_4[0]
            discrete_actions = precompute_soundspaces_actions(
                chosen_path, sound_sim.get_orientation()
            )
            budget = min(int(args.max_steps) - total_steps,
                         MAX_EPISODE_STEPS)
            if not discrete_actions:
                
                
                
                _, _, _, _, _, _, path_diag = follow_path(
                    envs, obs, ep, chosen_path, heading_xz, budget,
                    action_seq=[])
                path_diag.update({
                    "iteration": iteration,
                    "qwen_chosen_path": int(chosen_idx + 1) if chosen_idx is not None else 1,
                    "qwen_raw_response": choose_raw,
                    "candidate_targets_world": [x.tolist() for x in snapped_candidate_targets],
                    "termination": "predicted_waypoint_reached",
                })
                path_execution.append(path_diag)
                outputs = envs.step([ACTION_MAP["STOP"]])
                obs_l, _, dones, infos = [list(x) for x in zip(*outputs)]
                obs = obs_l[0]
                done = True
                final_info = infos[0]
                history_actions.append("STOP")
                total_steps += 1
                break
            obs, steps_used, done, step_info, acts, heading_xz, path_diag = follow_path(
                envs, obs, ep, chosen_path, heading_xz, budget,
                action_seq=discrete_actions)
            path_diag.update({
                "iteration": iteration,
                "qwen_chosen_path": int(chosen_idx + 1) if chosen_idx is not None else 1,
                "qwen_raw_response": choose_raw,
                "candidate_targets_world": [x.tolist() for x in snapped_candidate_targets],
            })
            path_execution.append(path_diag)
            total_steps += steps_used
            history_actions.extend(acts)
            final_info = step_info

            if done:
                break

            
            if "audiogoal" in obs:
                audio_wav2 = _write_temp_wav(np.asarray(obs["audiogoal"]), sample_rate, tag="recls")
                try:
                    qwen_top3, _ = fusion.run_qwen_top3_online(
                        omni_url=args.omni_url,
                        timeout_sec=float(args.request_timeout),
                        audio_file=audio_wav2,
                        candidate_classes=fusion.TASK_CATEGORIES_21,
                        max_new_tokens=int(args.max_new_tokens),
                        placeholder_image=placeholder_img,
                    )
                except Exception:
                    pass

            iteration += 1

        
        if not done:
            outputs = envs.step([ACTION_MAP["STOP"]])
            obs_l, _, dones, infos = [list(x) for x in zip(*outputs)]
            obs = obs_l[0]
            final_info = infos[0]
            history_actions.append("STOP")
            total_steps += 1

        success = float(final_info.get("success", 0.0))
        spl = float(final_info.get("spl", 0.0))
        soft_spl = float(final_info.get("soft_spl", 0.0))
        dtg = _extract_remaining_distance(final_info, -1.0)
        opt_n = _extract_optimal_num_actions(ep)
        sna = _extract_float_metric(final_info, ["sna", "success_weighted_by_num_actions"])
        if sna is None:
            sna = _compute_sna(success, opt_n, total_steps)
        sws_val = _extract_float_metric(final_info, ["sws", "success_when_silent"])
        if sws_val is None:
            is_silent = bool(getattr(ep, "is_silent", False)
                             or getattr(ep, "silent", False)
                             or is_silent_first)
            sws_val = float(success) if is_silent else None

        results.append({
            "scene": scene_name,
            "episode_id": ep_id,
            "sound_id": sound_id,
            "success": success,
            "spl": spl,
            "soft_spl": soft_spl,
            "sna": sna,
            "sws": sws_val,
            "final_dtg_m": dtg,
            "optimal_num_actions": opt_n,
            "num_steps": total_steps,
            "num_iterations": iteration + 1,
            "actions": history_actions,
            "qwen_top3_initial": qwen_top3,
            "path_execution": path_execution,
        })
        
        
        stats_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tag = "OK" if success > 0 else "FAIL"
        print(f"[{len(results)}/{num_target}] {scene_name} ep={ep_id} "
              f"{tag} sr={success:.0f} spl={spl:.3f} dtg={dtg:.3f}m "
              f"steps={total_steps} iters={iteration + 1}")

    envs.close()
    nav_cache.close()

    
    stats_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    sr_arr = [float(x["success"]) for x in results]
    spl_arr = [float(x["spl"]) for x in results]
    sna_arr = [float(x["sna"]) for x in results if x.get("sna") is not None]
    sws_arr = [float(x["sws"]) for x in results if x.get("sws") is not None]
    dtg_arr = [float(x["final_dtg_m"]) for x in results
               if x.get("final_dtg_m") is not None and x["final_dtg_m"] >= 0]
    qwen_top1_hits = [
        int(
            bool(x.get("qwen_top3_initial"))
            and str(x["qwen_top3_initial"][0]) == Path(str(x.get("sound_id", ""))).stem
        )
        for x in results
    ]
    total_available = _count_episodes(content_dir)

    summary = {
        "num_episodes": len(results),
        "protocol": {
            "dataset": "MP3D SemanticAudioNav v1 unheard-sound test",
            "max_episode_steps": int(args.max_steps),
            "success_distance_m": float(cfg.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE),
            "rgb_resolution": [
                int(cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH),
                int(cfg.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT),
            ],
            "spectrogram_shape": [65, 26, 2],
            "episode_sampling": (
                "seeded_random_subset_before_scene_grouping"
                if num_target < total_available else "complete_split"
            ),
            "sampling_seed": int(args.seed),
            "total_available_episodes": int(total_available),
            "num_scenes_evaluated": len({str(x["scene"]) for x in results}),
        },
        "localization_expert": "qwen_conditioned_spectrogram",
        "localization_checkpoint": str(Path(args.localization_ckpt).expanduser().resolve()),
        "SR": float(np.mean(sr_arr)) if sr_arr else 0.0,
        "SPL": float(np.mean(spl_arr)) if spl_arr else 0.0,
        "SNA": float(np.mean(sna_arr)) if sna_arr else None,
        "SWS": float(np.mean(sws_arr)) if sws_arr else None,
        "DTG_m": float(np.mean(dtg_arr)) if dtg_arr else None,
        "qwen_initial_top1_accuracy": (
            float(np.mean(qwen_top1_hits)) if qwen_top1_hits else None
        ),
        "SR_pct": (float(np.mean(sr_arr)) * 100) if sr_arr else 0.0,
        "SPL_pct": (float(np.mean(spl_arr)) * 100) if spl_arr else 0.0,
        "SNA_pct": (float(np.mean(sna_arr)) * 100) if sna_arr else None,
        "SWS_pct": (float(np.mean(sws_arr)) * 100) if sws_arr else None,
        "mean_steps": float(np.mean([x["num_steps"] for x in results])) if results else 0,
        "mean_iterations": float(np.mean([x["num_iterations"] for x in results])) if results else 0,
        "sna_valid_episodes": len(sna_arr),
        "sws_valid_episodes": len(sws_arr),
        "dtg_valid_episodes": len(dtg_arr),
        "stats_json": stats_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] stats  → {stats_path}")
    print(f"[DONE] summary → {summary_path}")


if __name__ == "__main__":
    main()

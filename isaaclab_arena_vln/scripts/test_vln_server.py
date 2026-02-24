#!/usr/bin/env python3
"""Minimal test client for the VLN VLM server.

Tests the Arena remote-policy ZeroMQ protocol:
  1. ping
  2. get_init_info (handshake)
  3. set_task_description
  4. get_action (with a dummy RGB image)
  5. reset

Usage (from another terminal on the same machine, or remotely):
    cd ~/my_scratch/Isaac/IsaacLab-Arena-vln-main4
    PYTHONPATH=. python isaaclab_arena_vln/scripts/test_vln_server.py --host localhost --port 5555
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from isaaclab_arena.remote_policy.message_serializer import MessageSerializer


def send_request(socket, endpoint: str, data: dict | None = None) -> dict:
    request = {"endpoint": endpoint}
    if data is not None:
        request["data"] = data
    raw = MessageSerializer.to_bytes(request)
    socket.send(raw)
    resp_raw = socket.recv()
    return MessageSerializer.from_bytes(resp_raw)


def main():
    parser = argparse.ArgumentParser("VLN Server Test Client")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=int, default=30000, help="ZMQ timeout (ms)")
    args = parser.parse_args()

    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout)
    sock.setsockopt(zmq.SNDTIMEO, args.timeout)
    addr = f"tcp://{args.host}:{args.port}"
    print(f"[test] Connecting to {addr} ...")
    sock.connect(addr)

    passed = 0
    failed = 0

    # ── Test 1: ping ──────────────────────────────────────────────
    print("\n[test 1/5] ping ...")
    try:
        resp = send_request(sock, "ping")
        assert resp.get("status") == "ok", f"Expected 'ok', got {resp}"
        print(f"  PASS: {resp}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Test 2: get_init_info (handshake) ─────────────────────────
    print("\n[test 2/5] get_init_info (handshake) ...")
    try:
        resp = send_request(sock, "get_init_info", {
            "requested_action_mode": "vln_velocity",
        })
        assert resp.get("status") == "success", f"Handshake failed: {resp}"
        config = resp["config"]
        print(f"  PASS: action_mode={config.get('action_mode')}, "
              f"action_dim={config.get('action_dim')}, "
              f"obs_keys={config.get('observation_keys')}, "
              f"default_duration={config.get('default_duration')}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Test 3: set_task_description ──────────────────────────────
    print("\n[test 3/5] set_task_description ...")
    try:
        instruction = "Walk through the hallway and turn left at the kitchen."
        resp = send_request(sock, "set_task_description", {
            "task_description": instruction,
        })
        print(f"  PASS: {resp}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Test 4: get_action (with dummy RGB image) ─────────────────
    print("\n[test 4/5] get_action (dummy 512x512 RGB image) ...")
    try:
        dummy_rgb = np.random.randint(0, 255, (1, 512, 512, 3), dtype=np.uint8)
        obs = {
            "camera_obs.robot_head_cam.rgb": dummy_rgb,
        }
        t0 = time.time()
        resp = send_request(sock, "get_action", {"observation": obs})
        dt = time.time() - t0
        action = resp.get("action")
        duration = resp.get("duration")
        vlm_text = resp.get("vlm_text", "N/A")
        if isinstance(action, np.ndarray):
            action = action.tolist()
        print(f"  PASS ({dt:.2f}s): action={action}, duration={duration}, vlm_text='{vlm_text}'")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Test 5: reset ─────────────────────────────────────────────
    print("\n[test 5/5] reset ...")
    try:
        resp = send_request(sock, "reset", {})
        print(f"  PASS: {resp}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*50}")

    sock.close()
    ctx.term()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

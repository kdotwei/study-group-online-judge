"""Exercise the real proxy template across private and shared Docker networks."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import pytest


def test_private_api_alias_avoids_shared_ingress_collision(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for the proxy network regression test")
    info = subprocess.run(
        ["docker", "info"], check=False, capture_output=True, timeout=10
    )
    if info.returncode:
        pytest.skip("Docker daemon is unavailable")

    root = Path(__file__).resolve().parents[1]
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("JUDGE_")
    }
    env.update(
        JUDGE_FLOCK_API_URL="http://flock.invalid",
        JUDGE_FLOCK_TOKEN="flock_cli_test-only",
    )
    config = json.loads(
        subprocess.check_output(
            [
                "docker",
                "compose",
                "--env-file",
                os.devnull,
                "-f",
                str(root / "compose.yaml"),
                "config",
                "--format",
                "json",
            ],
            env=env,
            text=True,
            timeout=15,
        )
    )
    image = next(
        line.split()[1]
        for line in (root / "ui/Dockerfile").read_text().splitlines()
        if line.endswith(" AS runtime")
    )
    backend = tmp_path / "backend.conf"
    backend.write_text(
        "server { listen 8000; location /api/leaderboard { "
        'default_type application/json; return 200 \'{"source":"judge"}\'; } }'
    )
    fixture = {
        "services": {
            # Only the explicit aliases come from production configuration. The
            # generic `api` name deliberately belongs to the unrelated service.
            "judge-backend": {
                "image": image,
                "volumes": [f"{backend}:/etc/nginx/conf.d/default.conf:ro"],
                "networks": {
                    "default": config["services"]["api"]["networks"]["default"]
                },
            },
            "unrelated-api": {
                "image": image,
                "networks": {"ingress": {"aliases": ["api"]}},
            },
            "proxy": {
                "image": image,
                "environment": config["services"]["leaderboard"]["environment"],
                "volumes": [
                    f"{root / 'ui/nginx.conf.template'}:/etc/nginx/templates/default.conf.template:ro"
                ],
                "networks": ["ingress", "default"],
            },
        },
        "networks": {"default": {}, "ingress": {}},
    }
    path = tmp_path / "compose.json"
    path.write_text(json.dumps(fixture))
    command = [
        "docker",
        "compose",
        "-p",
        f"judge-proxy-{uuid4().hex[:12]}",
        "-f",
        str(path),
    ]

    def compose(*args):
        return subprocess.run(
            [*command, *args], check=False, capture_output=True, text=True, timeout=120
        )

    try:
        started = compose("up", "-d", "--wait", "--wait-timeout", "30")
        assert started.returncode == 0, started.stderr
        deadline = monotonic() + 30
        while True:
            response = compose(
                "exec",
                "-T",
                "proxy",
                "wget",
                "-qO-",
                "-T",
                "2",
                "http://127.0.0.1/api/leaderboard",
            )
            if response.returncode == 0 or monotonic() >= deadline:
                break
            sleep(0.2)
        assert response.returncode == 0, response.stderr
        assert json.loads(response.stdout) == {"source": "judge"}
        wrong_backend = compose(
            "exec",
            "-T",
            "proxy",
            "wget",
            "-qO-",
            "-T",
            "2",
            "http://api:8000/api/leaderboard",
        )
        assert wrong_backend.returncode != 0
        assert "Connection refused" in wrong_backend.stderr
    finally:
        stopped = compose("down", "--volumes", "--remove-orphans")
        assert stopped.returncode == 0, stopped.stderr

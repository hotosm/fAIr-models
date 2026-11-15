"""Bring the local Compose stack up and register its ZenML stack.

Ships the Compose definition, DB init and ZenML stack config as
package resources so `fair stack up` works from any install without a repo
clone. Falls back to the repo layout for editable installs.
"""

import os
import shutil
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path

import httpx

ZENML_URL = "http://localhost:8080"


class FairStackError(RuntimeError):
    """A stack operation could not be completed."""


def _asset(package_rel: tuple[str, ...], repo_rel: tuple[str, ...]) -> Path:
    packaged = Path(str(files("fair").joinpath("infra", *package_rel)))
    if packaged.exists():
        return packaged
    repo_root = Path(__file__).resolve().parents[2]
    repo = repo_root.joinpath(*repo_rel)
    if repo.exists():
        return repo
    raise FairStackError(f"stack asset not found: {'/'.join(package_rel)}")


def compose_file() -> Path:
    return _asset(("compose", "docker-compose.yml"), ("infra", "compose", "docker-compose.yml"))


def stack_config_file() -> Path:
    return _asset(("stacks", "compose.yaml"), ("stacks", "compose.yaml"))


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FairStackError(f"{binary!r} not found on PATH; install it and retry")


def _compose(*args: str) -> None:
    _require("docker")
    subprocess.run(["docker", "compose", "-f", str(compose_file()), *args], check=True)


def _zenml_bin() -> str:
    return str(Path(sys.executable).parent / "zenml")


def _fetch_token(timeout_s: int = 300) -> str:
    login = f"{ZENML_URL}/api/v1/login"
    data = {"username": "default", "password": "", "grant_type": "password"}
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            resp = httpx.post(login, data=data, timeout=30)
            resp.raise_for_status()
            return resp.json()["access_token"]
        except httpx.HTTPError as err:
            if time.monotonic() >= deadline:
                raise FairStackError(f"ZenML unreachable at {ZENML_URL} after {timeout_s}s") from err
            time.sleep(10)


def _stack_exists(zenml: str, env: dict[str, str], name: str) -> bool:
    # `describe` exits non-zero when the stack is absent; that return code is the probe.
    probe = subprocess.run([zenml, "stack", "describe", name], env=env, capture_output=True)
    return probe.returncode == 0


def register_zenml_stack() -> None:
    zenml = _zenml_bin()
    env = {**os.environ, "ZENML_STORE_URL": ZENML_URL, "ZENML_STORE_API_TOKEN": _fetch_token()}
    subprocess.run([zenml, "init"], check=True, env=env, capture_output=True)
    subprocess.run([zenml, "stack", "set", "default"], check=True, env=env, capture_output=True)
    if _stack_exists(zenml, env, "compose"):
        subprocess.run([zenml, "stack", "delete", "compose", "-y", "-r"], check=True, env=env)
    config = str(stack_config_file())
    subprocess.run(
        [zenml, "stack", "import", "compose", "-f", config, "--ignore-version-mismatch"],
        check=True,
        env=env,
    )
    subprocess.run([zenml, "stack", "set", "compose"], check=True, env=env)


def up() -> None:
    _compose("up", "-d", "--wait")
    register_zenml_stack()


def down(volumes: bool = False) -> None:
    _compose("down", *(["-v"] if volumes else []))


def status() -> None:
    _compose("ps")

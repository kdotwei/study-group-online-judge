"""Server-side Flock reads for judged labs and the submission-free midterm."""

import json
import logging
import math
import os
import re
from datetime import UTC, datetime
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

MIDTERM_ID = "midterm"
logger = logging.getLogger("uvicorn.error.judge.leaderboard")


def finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


class FlockClient:
    def __init__(self):
        self.base = os.environ["JUDGE_FLOCK_API_URL"].rstrip("/")
        # Use a CLI token for REST reads. Old API keys are now wandb-only.
        # Keep the old variable name as an alias for deployments rotating in place.
        self.token = (
            os.getenv("JUDGE_FLOCK_TOKEN") or os.getenv("JUDGE_FLOCK_API_KEY") or ""
        ).strip()
        if not self.token.startswith("flock_cli_"):
            raise ValueError(
                "Set JUDGE_FLOCK_TOKEN to a Flock CLI token from flock login / "
                "flock token. W&B/API keys no longer work with Flock's REST API; "
                "JUDGE_FLOCK_API_KEY is only a deprecated alias for the CLI token."
            )
        self.web = (os.getenv("JUDGE_FLOCK_WEB_URL") or self.base).rstrip("/")
        self.entity = os.getenv("JUDGE_FLOCK_ENTITY") or "cerulean-labs"

    def get(self, path, **query):
        url = f"{self.base}/api/v1/{path}"
        if query:
            url += "?" + urlencode(query)
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "Cerulean-Study-Group-OJ/0.1",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 401:
                raise RuntimeError(
                    "Flock rejected JUDGE_FLOCK_TOKEN (HTTP 401). Run flock login "
                    "again and replace it with the CLI token from flock token."
                ) from error
            if error.code == 403:
                raise RuntimeError(
                    "Flock denied leaderboard access (HTTP 403). Check that the "
                    "CLI token's user can read the JUDGE_FLOCK_ENTITY organization."
                ) from error
            raise

    def project_path(self, project):
        return f"projects/{quote(self.entity, safe='')}/{quote(project, safe='')}"

    def project_url(self, project):
        return f"{self.web}/{quote(self.entity, safe='')}/{quote(project, safe='')}"

    def avatar_url(self, path):
        # Flock serves these images publicly through its dashboard too. Use that
        # browser-facing origin rather than the possibly internal API address.
        if isinstance(path, str) and re.fullmatch(
            r"/api/v1/avatars/[0-9a-f]{64}", path
        ):
            return f"{self.web}{path}"
        return None

    def runs(self, project, filters):
        offset = 0
        while True:
            try:
                page = self.get(
                    f"{self.project_path(project)}/runs",
                    columns="config.*,summary.*,created_at,owner,state",
                    filter=json.dumps(filters),
                    sort="created_at,name",
                    limit=1000,
                    offset=offset,
                )
            except HTTPError as error:
                if error.code != 404 or offset:
                    raise
                logger.warning(
                    "Flock project not found: %s/%s; showing an empty board",
                    self.entity,
                    project,
                )
                return
            ids = page["ids"]
            for index, run_id in enumerate(ids):
                yield {
                    "id": run_id,
                    **{key: values[index] for key, values in page["data"].items()},
                }
            offset += len(ids)
            if offset >= page["total"]:
                return
            if not ids:
                raise ValueError(
                    "Flock returned an empty page before the end of the run table"
                )

    def exceeds_memory_limit(self, project, run_id):
        # Fetch every system sample: downsampling can hide a brief VRAM spike.
        history = self.get(
            f"{self.project_path(project)}/runs/{quote(run_id, safe='')}/history",
            stream="system",
            max_points=0,
        )
        for key, curve in history["series"].items():
            # W&B reports allocated GPU memory in bytes.
            if re.fullmatch(r"system\.gpu\.\d+\.memoryAllocatedBytes", key) and any(
                finite(value) and value > 80 * 1024**3 for value in curve["y"]
            ):
                return True
        return False


def submitted_at(value):
    if finite(value):
        return datetime.fromtimestamp(value, UTC).isoformat()
    parsed = datetime.fromisoformat(str(value))
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    ).isoformat()


def flock_submissions(task_ids):
    client = FlockClient()
    labs = os.getenv("JUDGE_FLOCK_LABS_PROJECT") or "study-group-labs"
    midterm = os.getenv("JUDGE_FLOCK_MIDTERM_PROJECT") or "midterm-nanogpt"
    # Verify organization access before treating a missing project as empty.
    members = client.get(f"orgs/{quote(client.entity, safe='')}/members")["members"]
    owners = {member["id"]: member for member in members}
    result = []
    for run in client.runs(labs, {"summary.judge_status": "completed"}):
        actor = run.get("config.github_actor")
        task = run.get("config.task_id")
        if task not in task_ids or not isinstance(actor, str) or not actor.strip():
            continue
        try:
            created = submitted_at(
                run.get("config.submitted_at") or run.get("created_at")
            )
        except ValueError, TypeError, OverflowError, OSError:
            continue
        result.append(
            {
                "id": run["id"],
                "task_id": task,
                "github_actor": actor.strip(),
                "submitted_at": created,
                "passed": run.get("summary.passed"),
                "metrics": {
                    key.removeprefix("summary."): value
                    for key, value in run.items()
                    if key.startswith("summary.")
                },
                "run_url": f"{client.project_url(labs)}/runs/{quote(run['id'], safe='')}",
                "project_url": client.project_url(labs),
            }
        )
    for run in client.runs(midterm, {"state": "finished"}):
        steps, loss = run.get("summary.total_train_steps"), run.get("summary.val_loss")
        owner = run.get("owner")
        if (
            not finite(steps)
            or steps <= 0
            or int(steps) != steps
            or not finite(loss)
            or not 0 <= loss < 3.28
            or owner not in owners
        ):
            continue
        try:
            created = submitted_at(run.get("created_at"))
        except ValueError, TypeError, OverflowError, OSError:
            continue
        if client.exceeds_memory_limit(midterm, run["id"]):
            logger.info(
                "Midterm run excluded: logged GPU memory exceeds 80 GiB run=%s",
                run["id"],
            )
            continue
        result.append(
            {
                "id": run["id"],
                "task_id": MIDTERM_ID,
                "github_actor": owners[owner]["name"],
                "participant_id": f"flock:{owner}",
                "avatar_url": client.avatar_url(owners[owner].get("avatar_url")),
                "submitted_at": created,
                "passed": True,
                "metrics": {"total_train_steps": int(steps), "val_loss": loss},
                "run_url": f"{client.project_url(midterm)}/runs/{quote(run['id'], safe='')}",
                "project_url": client.project_url(midterm),
            }
        )
    logger.info(
        "Flock runs fetched labs=%s midterm=%s accepted=%d", labs, midterm, len(result)
    )
    return result

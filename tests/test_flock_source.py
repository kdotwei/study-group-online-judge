import json
from email.message import Message
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from judge.flock_source import FlockClient, flock_submissions
from judge.leaderboard import LeaderboardService, rank_submissions, safe_traceback
from judge.tasks import TASKS


@pytest.fixture(autouse=True)
def flock_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_FLOCK_API_URL", "https://api.flock.test")
    monkeypatch.setenv("JUDGE_FLOCK_TOKEN", "flock_cli_private-test-token")
    monkeypatch.delenv("JUDGE_FLOCK_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_FLOCK_WEB_URL", "https://flock.test")
    monkeypatch.setenv("JUDGE_FLOCK_ENTITY", "cerulean-labs")
    monkeypatch.setenv("JUDGE_FLOCK_LABS_PROJECT", "study-group-labs")
    monkeypatch.setenv("JUDGE_FLOCK_MIDTERM_PROJECT", "midterm-nanogpt")


def midterm(run_id="one", owner=7, steps=100, loss=3.27):
    return {
        "id": run_id,
        "owner": owner,
        "created_at": 1792400000,
        "summary.total_train_steps": steps,
        "summary.val_loss": loss,
    }


def fetch(labs=(), runs=(), members=None):
    if members is None:
        members = [{"id": 7, "name": "Alice"}, {"id": 8, "name": "Alice"}]
    with (
        patch.object(FlockClient, "runs", side_effect=[labs, runs]),
        patch.object(FlockClient, "get", return_value={"members": members}),
        patch.object(FlockClient, "exceeds_memory_limit", return_value=False),
    ):
        return flock_submissions(set(TASKS))


def test_midterm_best_per_owner_and_duplicate_display_names():
    items = fetch(
        runs=[
            midterm(),
            midterm("better", steps=80),
            midterm("other", owner=8, steps=90),
        ]
    )
    board = next(board for board in rank_submissions(items) if board["id"] == "midterm")
    assert board["primary_metric"] == "total_train_steps"
    assert board["metric_direction"] == "minimize"
    assert [entry["score"] for entry in board["entries"]] == [80, 90]
    assert [entry["participant_id"] for entry in board["entries"]] == [
        "flock:7",
        "flock:8",
    ]
    assert board["entries"][0]["val_loss"] == 3.27
    assert board["entries"][0]["attempts"] == 2
    assert (
        board["entries"][0]["run_url"]
        == "https://flock.test/cerulean-labs/midterm-nanogpt/runs/better"
    )
    assert "midterm" not in TASKS


def test_midterm_profiles_follow_owner_id_through_ranking():
    paths = [f"/api/v1/avatars/{digest * 64}" for digest in ("a", "b")]
    members = [
        {"id": 7, "name": "Alice", "avatar_url": paths[0], "email": "private@test"},
        {"id": 8, "name": "Alice", "avatar_url": paths[1]},
    ]
    items = fetch(runs=[midterm(), midterm("other", owner=8)], members=members)
    board = next(board for board in rank_submissions(items) if board["id"] == "midterm")
    assert [entry["avatar_url"] for entry in board["entries"]] == [
        f"https://flock.test{path}" for path in paths
    ]
    assert [entry["participant_id"] for entry in board["entries"]] == [
        "flock:7",
        "flock:8",
    ]
    assert "private@test" not in json.dumps(items)
    assert "flock_cli_private-test-token" not in json.dumps(board)


@pytest.mark.parametrize(
    "avatar", [None, "", "https://other.test/image", "//other.test/image"]
)
def test_missing_or_invalid_flock_avatar_keeps_participant(avatar):
    items = fetch(
        runs=[midterm()], members=[{"id": 7, "name": "Alice", "avatar_url": avatar}]
    )
    assert len(items) == 1
    assert items[0]["avatar_url"] is None


def test_avatar_defaults_to_api_origin_without_dashboard_url(monkeypatch):
    monkeypatch.delenv("JUDGE_FLOCK_WEB_URL")
    path = "/api/v1/avatars/" + "a" * 64
    assert FlockClient().avatar_url(path) == f"https://api.flock.test{path}"


@pytest.mark.parametrize(
    "steps,loss",
    [
        (None, 3.2),
        (True, 3.2),
        (0, 3.2),
        (-1, 3.2),
        (2.5, 3.2),
        (float("inf"), 3.2),
        ("100", 3.2),
        (100, None),
        (100, True),
        (100, float("nan")),
        (100, float("inf")),
        (100, -1),
        (100, 3.28),
        (100, 3.4),
    ],
)
def test_nonqualifying_summaries_are_excluded(steps, loss):
    assert fetch(runs=[midterm(steps=steps, loss=loss)]) == []


def test_unknown_owner_and_invalid_timestamp_are_excluded():
    bad = midterm()
    bad["created_at"] = "yesterday"
    assert fetch(runs=[midterm(owner=None), midterm(owner=999), bad]) == []


def test_lab_adapter_preserves_judge_metadata():
    run = {
        "id": "lab-run",
        "config.github_actor": "alice",
        "config.task_id": "lab4",
        "config.submitted_at": "2026-10-01T00:00:00Z",
        "created_at": 1792400000,
        "summary.passed": True,
        "summary.score": 2.3,
    }
    items = fetch(labs=[run, {**run, "config.task_id": "unknown"}])
    assert len(items) == 1
    assert items[0]["submitted_at"] == "2026-10-01T00:00:00+00:00"
    assert items[0]["metrics"]["score"] == 2.3
    assert items[0]["run_url"].endswith("/study-group-labs/runs/lab-run")
    assert "avatar_url" not in items[0]


def test_pagination_column_orientation_and_filtering():
    def page(ids, offset):
        return {
            "total": 3,
            "ids": ids,
            "data": {"owner": [7] * len(ids)},
            "offset": offset,
        }

    with patch.object(
        FlockClient, "get", side_effect=[page(["a", "b"], 0), page(["c"], 2)]
    ) as get:
        rows = list(FlockClient().runs("midterm-nanogpt", {"state": "finished"}))
    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert get.call_args.kwargs["offset"] == 2
    assert json.loads(get.call_args.kwargs["filter"]) == {"state": "finished"}


@pytest.mark.parametrize("legacy_key", [None, "flock-old-key"])
def test_request_credentials_stay_on_server(monkeypatch, legacy_key):
    if legacy_key:
        monkeypatch.setenv("JUDGE_FLOCK_API_KEY", legacy_key)
    response = MagicMock()
    response.read.return_value = b'{"members": []}'
    with patch("judge.flock_source.urlopen") as send:
        send.return_value.__enter__.return_value = response
        assert FlockClient().get("orgs/cerulean-labs/members") == {"members": []}
    request = send.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer flock_cli_private-test-token"
    assert request.get_header("User-agent") == "Cerulean-Study-Group-OJ/0.1"
    assert request.get_header("Accept") == "application/json"
    assert "flock_cli_private-test-token" not in request.full_url
    assert "flock_cli_private-test-token" not in safe_traceback(
        ValueError("flock_cli_private-test-token")
    )


@pytest.mark.parametrize("token", [None, "", "flock-old-key", "flock-wandb-new-key"])
@pytest.mark.parametrize("variable", ["JUDGE_FLOCK_TOKEN", "JUDGE_FLOCK_API_KEY"])
def test_missing_or_wandb_credentials_explain_migration(monkeypatch, token, variable):
    monkeypatch.delenv("JUDGE_FLOCK_TOKEN")
    if token is not None:
        monkeypatch.setenv(variable, token)
    with (
        patch("judge.flock_source.urlopen") as send,
        pytest.raises(ValueError, match="Set JUDGE_FLOCK_TOKEN.*CLI token") as error,
    ):
        FlockClient()
    send.assert_not_called()
    if token:
        assert token not in str(error.value)


@pytest.mark.parametrize("new_value", [None, ""])
def test_legacy_variable_accepts_cli_token(monkeypatch, new_value):
    if new_value is None:
        monkeypatch.delenv("JUDGE_FLOCK_TOKEN")
    else:
        monkeypatch.setenv("JUDGE_FLOCK_TOKEN", new_value)
    monkeypatch.setenv("JUDGE_FLOCK_API_KEY", "flock_cli_legacy-variable-token")
    response = MagicMock()
    response.read.return_value = b'{"members": []}'
    with patch("judge.flock_source.urlopen") as send:
        send.return_value.__enter__.return_value = response
        FlockClient().get("orgs/cerulean-labs/members")
    assert send.call_args.args[0].get_header("Authorization") == (
        "Bearer flock_cli_legacy-variable-token"
    )
    assert "flock_cli_legacy-variable-token" not in safe_traceback(
        ValueError("flock_cli_legacy-variable-token")
    )


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_preserves_standings_and_logs_recovery(
    tmp_path, monkeypatch, caplog, status
):
    monkeypatch.delenv("JUDGE_SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("JUDGE_SLACK_BOT_TOKEN", raising=False)
    service = LeaderboardService(tmp_path / "judge.db")
    # Exercise members, both run lists, and system history through the HTTP client.
    responses = [
        {"members": [{"id": 7, "name": "Alice"}]},
        {"total": 0, "ids": [], "data": {}},
        {
            "total": 1,
            "ids": ["one"],
            "data": {key: [value] for key, value in midterm().items() if key != "id"},
        },
        {"series": {}},
    ]
    with patch("judge.flock_source.urlopen") as send:
        replies = []
        for body in responses:
            response = MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                body
            ).encode()
            replies.append(response)
        send.side_effect = replies
        service.refresh()
        assert send.call_count == 4
        assert all(
            call.args[0].get_header("Authorization")
            == "Bearer flock_cli_private-test-token"
            for call in send.call_args_list
        )
        assert send.call_args.args[0].full_url.endswith("stream=system&max_points=0")
        assert service.snapshot["error"] is None
        boards = {board["id"]: board for board in service.snapshot["labs"]}
        assert boards["midterm"]["entries"][0]["score"] == 100
        previous = service.snapshot
        send.side_effect = HTTPError(
            "https://api.flock.test/api/v1/orgs/cerulean-labs/members",
            status,
            "Unauthorized",
            Message(),
            None,
        )
        service.refresh()
    assert service.snapshot["labs"] == previous["labs"]
    assert service.snapshot["updated_at"] == previous["updated_at"]
    assert service.snapshot["error"] is not None
    assert f"HTTP {status}" in caplog.text
    assert ("flock login" if status == 401 else "JUDGE_FLOCK_ENTITY") in caplog.text
    assert "flock_cli_private-test-token" not in caplog.text
    assert "JUDGE_FLOCK_TOKEN" not in service.snapshot["error"]


def test_other_http_errors_propagate():
    error = HTTPError("https://api.flock.test", 500, "failed", Message(), None)
    with (
        patch("judge.flock_source.urlopen", side_effect=error),
        pytest.raises(HTTPError) as caught,
    ):
        FlockClient().get("orgs/cerulean-labs/members")
    assert caught.value is error


@pytest.mark.parametrize(
    "peak,excluded", [(80 * 1024**3, False), (80 * 1024**3 + 1, True), (None, False)]
)
def test_vram_checks_all_samples_without_downsampling(peak, excluded):
    with patch.object(
        FlockClient,
        "get",
        return_value={
            "series": {
                "system.gpu.3.memoryAllocatedBytes": {"y": [2000, peak, 1000]},
                "system.memory": {"y": [999999]},
            }
        },
    ) as get:
        assert FlockClient().exceeds_memory_limit("midterm-nanogpt", "run") is excluded
    assert get.call_args.kwargs == {"stream": "system", "max_points": 0}


def test_over_memory_run_is_excluded():
    with (
        patch.object(FlockClient, "runs", side_effect=[[], [midterm()]]),
        patch.object(
            FlockClient, "get", return_value={"members": [{"id": 7, "name": "Alice"}]}
        ),
        patch.object(FlockClient, "exceeds_memory_limit", return_value=True),
    ):
        assert flock_submissions(set(TASKS)) == []


def test_missing_midterm_keeps_existing_lab_results():
    lab_page = {
        "total": 1,
        "ids": ["lab-run"],
        "data": {
            "config.task_id": ["lab4"],
            "config.github_actor": ["alice"],
            "created_at": [1792400000],
            "summary.score": [2.3],
            "summary.passed": [True],
        },
    }
    with patch.object(
        FlockClient,
        "get",
        side_effect=[
            {"members": []},
            lab_page,
            HTTPError("https://api.flock.test", 404, "missing", Message(), None),
        ],
    ):
        items = flock_submissions(set(TASKS))
    boards = {board["id"]: board for board in rank_submissions(items)}
    assert boards["lab4"]["entries"][0]["score"] == 2.3
    assert boards["midterm"]["entries"] == []


@pytest.mark.parametrize("status", [401, 403, 500])
def test_project_read_failures_still_propagate(status):
    with (
        patch.object(
            FlockClient,
            "get",
            side_effect=HTTPError(
                "https://api.flock.test", status, "failed", Message(), None
            ),
        ),
        pytest.raises(HTTPError),
    ):
        list(FlockClient().runs("midterm-nanogpt", {}))


def test_missing_organization_is_not_treated_as_empty():
    with (
        patch.object(
            FlockClient,
            "get",
            side_effect=HTTPError(
                "https://api.flock.test", 404, "missing org", Message(), None
            ),
        ),
        pytest.raises(HTTPError),
    ):
        flock_submissions(set(TASKS))


def test_project_disappearing_during_pagination_is_not_partial_success():
    with (
        patch.object(
            FlockClient,
            "get",
            side_effect=[
                {"total": 2, "ids": ["run"], "data": {}},
                HTTPError("https://api.flock.test", 404, "deleted", Message(), None),
            ],
        ),
        pytest.raises(HTTPError),
    ):
        list(FlockClient().runs("midterm-nanogpt", {}))

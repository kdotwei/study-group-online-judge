"""Exercise the built leaderboard against local Flock-shaped snapshots."""

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from playwright.sync_api import expect, sync_playwright

from judge.leaderboard import rank_submissions

pytestmark = pytest.mark.screenshot


@pytest.mark.parametrize("width", [1280, 390])
def test_midterm_navigation_font_and_refresh(tmp_path, width):
    dist = Path(__file__).resolve().parents[1] / "ui/dist"
    entries = [
        {
            "id": "run-a",
            "task_id": "midterm",
            "github_actor": "Alice",
            "participant_id": "flock:7",
            "avatar_url": "https://flock.test/api/v1/avatars/" + "a" * 64,
            "passed": True,
            "metrics": {"total_train_steps": 100, "val_loss": 3.27},
            "submitted_at": "2026-10-20T00:00:00Z",
            "project_url": "https://flock.test/cerulean-labs/midterm-nanogpt",
            "run_url": "https://flock.test/cerulean-labs/midterm-nanogpt/runs/run-a",
        },
        {
            "id": "run-b",
            "task_id": "midterm",
            "github_actor": "Bob",
            "participant_id": "flock:8",
            "passed": True,
            "metrics": {"total_train_steps": 120, "val_loss": 3.26},
            "submitted_at": "2026-10-20T01:00:00Z",
            "run_url": None,
        },
    ]
    boards = rank_submissions(entries)
    boards.insert(0, {**boards[0], "id": "lab10", "entries": []})
    snapshot = {
        "source": "flock",
        "updated_at": "2026-10-20T07:00:00Z",
        "error": None,
        "labs": boards,
    }
    loads = []

    def serve(route):
        path = unquote(urlparse(route.request.url).path)
        if path == "/api/leaderboard":
            loads.append(True)
            route.fulfill(content_type="application/json", body=json.dumps(snapshot))
            return
        asset = (dist / (path.lstrip("/") or "index.html")).resolve()
        assert asset.is_relative_to(dist)
        route.fulfill(path=str(asset))

    with sync_playwright() as playwright, playwright.chromium.launch() as browser:
        page = browser.new_page(viewport={"width": width, "height": 1000})
        page.route("http://leaderboard.invalid/**", serve)
        avatar_requests = []

        def avatar(route):
            avatar_requests.append(route.request)
            if route.request.url.endswith("b" * 64):
                route.fulfill(status=404)
            else:
                route.fulfill(
                    content_type="image/svg+xml",
                    body='<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36"><rect width="36" height="36" fill="blue"/></svg>',
                )

        page.route("https://flock.test/api/v1/avatars/*", avatar)
        github_requests = []

        def github(route):
            github_requests.append(route.request.url)
            route.fulfill(status=404)

        page.route("https://github.com/**", github)
        page.clock.install()
        page.goto("http://leaderboard.invalid/", wait_until="networkidle")
        expect(page.locator(".board h2")).to_have_text("Speedrunning NanoGPT")
        buttons = page.locator(".lab-button strong").all_text_contents()
        assert buttons[:3] == ["Midterm", "Lab 10", "Lab 5"]
        assert buttons[-1] == "Lab 1"
        assert page.locator(".lab-button.active strong").inner_text() == "Midterm"
        expect(page.get_by_role("columnheader", name="Validation loss")).to_be_visible()
        expect(page.locator("tbody tr")).to_have_count(2)
        expect(page.locator(".avatar img")).to_have_count(1)
        expect(page.locator(".avatar img")).to_have_attribute(
            "src", entries[0]["avatar_url"]
        )
        assert page.locator(".avatar img").evaluate("img => img.naturalWidth") == 36
        expect(page.locator(".avatar").nth(1)).to_have_text("BO")
        assert "authorization" not in avatar_requests[0].headers
        assert (
            page.locator(".nav-link").get_attribute("href") == entries[0]["project_url"]
        )
        assert page.locator(".run-link").get_attribute("href") == entries[0]["run_url"]
        assert page.locator(".hero h1").inner_text() in json.loads(
            (dist.parent / "src/greetings.json").read_text()
        )
        for selector in [".brand", ".hero h1", ".board h2", ".search input", ".score"]:
            assert "Host Grotesk" in page.locator(selector).first.evaluate(
                "el => getComputedStyle(el).fontFamily"
            )
        assert page.evaluate("document.fonts.check('400 16px \"Host Grotesk\"')")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(tmp_path / f"midterm-{width}.png"), full_page=True)

        # A failed image falls back to initials; a new digest on refresh retries.
        alice = next(board for board in boards if board["id"] == "midterm")["entries"][
            0
        ]
        alice["avatar_url"] = "https://flock.test/api/v1/avatars/" + "b" * 64
        page.clock.fast_forward(30_000)
        expect(page.locator(".avatar img")).to_have_count(0)
        expect(page.locator(".avatar").first).to_have_text("AL")
        alice["avatar_url"] = "https://flock.test/api/v1/avatars/" + "c" * 64
        page.clock.fast_forward(30_000)
        expect(page.locator(".avatar img")).to_have_count(1)
        expect(page.locator(".avatar img")).to_have_attribute(
            "src", alice["avatar_url"]
        )
        alice["avatar_url"] = None
        page.clock.fast_forward(30_000)
        expect(page.locator(".avatar img")).to_have_count(0)
        assert not github_requests

        page.get_by_role("textbox", name="Search participants").fill("alice")
        expect(page.locator("tbody tr")).to_have_count(1)
        page.get_by_role("button", name="10 Lab 10").click()
        expect(page.locator(".lab-button.active strong")).to_have_text("Lab 10")
        assert (
            page.get_by_role("textbox", name="Search participants").input_value() == ""
        )
        page.clock.fast_forward(30_000)
        expect(page.locator(".lab-button.active strong")).to_have_text("Lab 10")
        assert len(loads) >= 2

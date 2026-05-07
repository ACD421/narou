from unittest.mock import MagicMock, patch

import httpx

from narou.ingestion import GreenhouseAdapter, LeverAdapter


GH_SAMPLE = {
    "jobs": [
        {
            "id": 1001,
            "title": "Senior Security Engineer",
            "absolute_url": "https://boards.greenhouse.io/fake/jobs/1001",
            "location": {"name": "Remote - USA"},
            "departments": [{"name": "Security"}],
            "content": "<p>We are hiring a <strong>Senior</strong> Security Engineer. &amp; more.</p>",
            "updated_at": "2026-04-01T00:00:00Z",
            "first_published": "2026-03-15T00:00:00Z",
        },
        {
            "id": 1002,
            "title": "Broken - no content",
            "absolute_url": None,
            "location": None,
            "departments": [],
            "updated_at": "2026-04-01T00:00:00Z",
        },
    ]
}


def _make_response(status_code=200, json_body=None, raise_error=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if raise_error:
        resp.json.side_effect = raise_error
    return resp


def test_greenhouse_parses_jobs_and_strips_html():
    client = MagicMock()
    client.get.return_value = _make_response(json_body=GH_SAMPLE)
    adapter = GreenhouseAdapter(client=client)
    result = adapter.fetch("fake")
    assert result.ok
    assert result.count == 2
    first = result.jobs[0]
    assert first.title == "Senior Security Engineer"
    assert "<" not in first.description
    assert "&amp;" not in first.description
    assert first.location == "Remote - USA"
    assert first.department == "Security"
    assert first.posted_at is not None


def test_greenhouse_404_returns_error():
    client = MagicMock()
    client.get.return_value = _make_response(status_code=404)
    adapter = GreenhouseAdapter(client=client)
    result = adapter.fetch("missing")
    assert not result.ok
    assert "not found" in result.error


def test_greenhouse_network_failure_returns_error():
    client = MagicMock()
    client.get.side_effect = httpx.ConnectError("no route")
    adapter = GreenhouseAdapter(client=client)
    result = adapter.fetch("boom")
    assert not result.ok
    assert "network" in result.error


LEVER_SAMPLE = [
    {
        "id": "abc-123",
        "text": "Research Engineer, ML",
        "categories": {"location": "San Francisco", "team": "Research"},
        "descriptionPlain": "Research role on our ML team.",
        "hostedUrl": "https://jobs.lever.co/fake/abc-123",
        "createdAt": 1_700_000_000_000,
    }
]


def test_lever_parses_jobs():
    client = MagicMock()
    client.get.return_value = _make_response(json_body=LEVER_SAMPLE)
    adapter = LeverAdapter(client=client)
    result = adapter.fetch("fake")
    assert result.ok
    assert result.count == 1
    j = result.jobs[0]
    assert j.title == "Research Engineer, ML"
    assert j.location == "San Francisco"
    assert j.department == "Research"
    assert j.posted_at is not None

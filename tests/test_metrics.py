import json

from laya_doom.metrics import load_jsonl, summarize_episodes


def test_load_jsonl_roundtrip(tmp_path):
    path = tmp_path / "episodes.jsonl"
    rows = [{"episode": 0, "steps": 10, "died": False}, {"episode": 1, "steps": 20, "died": True}]
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    assert load_jsonl(path) == rows


def test_summarize_episodes_aggregates():
    rows = [
        {
            "steps": 10,
            "survival_tics": 100,
            "distance_travelled": 50.0,
            "kills": 1,
            "damage_given": 30,
            "damage_taken": 10,
            "health_remaining": 90,
            "ammo_used": 5,
            "items_collected": 1,
            "total_reward": 5.0,
            "died": False,
            "completed": True,
            "mean_confidence": 0.8,
            "mean_latency_ms": 12.0,
            "action_counts": {"move_forward": 3, "attack": 2},
        },
        {
            "steps": 20,
            "survival_tics": 200,
            "distance_travelled": 100.0,
            "kills": 3,
            "damage_given": 60,
            "damage_taken": 40,
            "health_remaining": 0,
            "ammo_used": 15,
            "items_collected": 0,
            "total_reward": 15.0,
            "died": True,
            "completed": False,
            "mean_confidence": 0.6,
            "mean_latency_ms": 20.0,
            "action_counts": {"move_forward": 5, "attack": 4},
        },
    ]
    summary = summarize_episodes(rows)
    assert summary["episodes"] == 2
    assert summary["mean_steps"] == 15
    assert summary["death_rate"] == 0.5
    assert summary["completion_rate"] == 0.5
    assert summary["mean_kills"] == 2
    assert summary["action_distribution"] == {"move_forward": 8, "attack": 6}


def test_summarize_episodes_empty():
    assert summarize_episodes([]) == {"episodes": 0}

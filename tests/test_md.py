from transcribe.md import format_ts, json_to_markdown, duration_seconds


def test_format_ts():
    assert format_ts(0) == "00:00:00"
    assert format_ts(65) == "00:01:05"
    assert format_ts(3661) == "01:01:01"
    assert format_ts(-3) == "00:00:00"
    assert format_ts(None) == "00:00:00"


def test_duration_from_segments():
    data = {
        "segments": [
            {"start": 0, "end": 10, "text": "hello"},
            {"start": 10, "end": 125.4, "text": "world"},
        ]
    }
    assert duration_seconds(data) == 125.4


def test_json_to_markdown_lines():
    data = {
        "duration": 90,
        "segments": [
            {"start": 0, "end": 5, "text": " Opening remarks "},
            {"start": 5, "end": 12, "text": ""},
            {"start": 12.2, "end": 20, "text": "Volume buzz"},
        ],
    }
    md = json_to_markdown(
        data,
        source_name="lecture.mp4",
        model="mlx-community/whisper-large-v3-turbo",
    )
    assert md.startswith("# Transcript — lecture.mp4")
    assert "`mlx-community/whisper-large-v3-turbo`" in md
    assert "[00:00:00] Opening remarks" in md
    assert "[00:00:12] Volume buzz" in md
    assert md.count("[00:") == 2  # empty segment skipped

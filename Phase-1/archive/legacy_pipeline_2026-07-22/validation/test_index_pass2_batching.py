from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from index.build import _cluster_assignments


class FakeVectorizer:
    def transform(self, texts):
        return list(texts)


class FakeKMeans:
    def predict(self, matrix):
        return [len(text) % 3 for text in matrix]


def main() -> int:
    chunks = [
        {"chunk_uid": "a", "text": "aaaa"},
        {"chunk_uid": "b", "text": "bb"},
        {"chunk_uid": "c", "text": "ccc"},
    ]
    assignments = _cluster_assignments(FakeVectorizer(), FakeKMeans(), chunks)
    assert assignments == [(1, "a"), (2, "b"), (0, "c")]
    assert _cluster_assignments(FakeVectorizer(), FakeKMeans(), []) == []
    print("[index-pass2-batching] batch cluster assignment preserves uid order: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import hashlib
from pathlib import Path

import pytest

from docguard.discovery.fetch import Snapshot
from docguard.knowledge.pipeline import KnowledgeBase
from docguard.processing.config import load_chunking_config
from docguard.processing.token_budget import TokenBudget

ROOT = Path(__file__).resolve().parents[2]
PAGES = {
    "https://redis.io/docs/persistence/": b'<title>Redis</title><main><h1 id="rdb">RDB snapshots</h1><p>Redis RDB creates point-in-time snapshots. Configure save intervals to enable snapshots.</p><h2 id="aof">AOF</h2><p>Append only files record write operations.</p></main>',
    "https://docs.python.org/3/library/configparser.html": b"<title>Python</title><article><h1>ConfigParser</h1><p>ConfigParser reads configuration files in INI format. Use read to load a file.</p></article>",
}


class FixtureFetcher:
    def fetch(self, url):
        body = PAGES[url]
        return Snapshot(
            url,
            url,
            "2026-10-27T00:00:00+00:00",
            hashlib.sha256(body).hexdigest(),
            body,
            url + "robots.txt",
            404,
            hashlib.sha256(b"").hexdigest(),
        )


class TinyEmbedder:
    async def embed_passages(self, texts):
        return [[float("Redis" in t), float("Python" in t), 0.1] for t in texts]

    async def embed_queries(self, texts):
        return [[float("Redis" in t), float("Python" in t), 0.1] for t in texts]


class TinyReranker:
    def score_pairs(self, query, passages):
        from docguard.retrieval.reranker import PairScore

        return [
            PairScore(0.9 if "Append only" in text else -0.3, False)
            for text in passages
        ]


@pytest.mark.asyncio
async def test_full_retrieval_preserves_real_route_scores_and_rerank_order(tmp_path):
    base = make_base(
        tmp_path,
        embedder=TinyEmbedder(),
        embedding_id="tiny-v1",
        reranker=TinyReranker(),
        reranker_id="tiny-reranker-v1",
        default_method="full",
    )
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    await base.import_urls(
        "python-other", [list(PAGES)[1]], product="Python", fetcher=FixtureFetcher()
    )
    result = await base.answer("redis", "save intervals")
    assert result["retrieval"]["method"] == "full"
    assert result["refusal_calibration"] == "not_calibrated_for_generic_documents"
    assert result["decision"]["confidence"] is None
    first = result["evidence"][0]
    assert "Append only" in first["text"]
    assert first["provenance"]["rerank_score"] == 0.9
    assert first["provenance"]["dense_rank"] is not None
    assert first["provenance"]["bm25_rank"] is None
    assert first["provenance"]["rrf_score"] > 0
    assert first["provenance"]["original_rrf_rank"] is not None
    assert all(e["knowledge_base_id"] == "redis" for e in result["evidence"])


@pytest.mark.asyncio
async def test_full_query_upgrades_legacy_snapshot_without_refetch_or_rechunk(tmp_path):
    legacy = make_base(tmp_path)
    old = await legacy.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    old_dir = tmp_path / "redis" / old["snapshot_id"]
    original_chunks = (old_dir / "chunks.json").read_bytes()
    base = make_base(
        tmp_path,
        embedder=TinyEmbedder(),
        embedding_id="tiny-v1",
        reranker=TinyReranker(),
        reranker_id="tiny-reranker-v1",
        default_method="full",
    )
    result = await base.answer("redis", "RDB")
    new = base.info("redis")
    assert new["snapshot_id"] != old["snapshot_id"]
    assert new["derived_from_snapshot"] == old["snapshot_id"]
    assert (
        tmp_path / "redis" / new["snapshot_id"] / "chunks.json"
    ).read_bytes() == original_chunks
    assert (old_dir / "chunks.json").read_bytes() == original_chunks
    assert result["retrieval"]["snapshot_id"] == new["snapshot_id"]
    await base.answer("redis", "RDB")
    assert base.info("redis")["snapshot_id"] == new["snapshot_id"]


@pytest.mark.asyncio
async def test_failed_dense_upgrade_keeps_active_snapshot(tmp_path):
    legacy = make_base(tmp_path)
    old = await legacy.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )

    class Broken(TinyEmbedder):
        async def embed_passages(self, texts):
            raise RuntimeError("fixture embedding failure")

    base = make_base(
        tmp_path,
        embedder=Broken(),
        embedding_id="tiny-v1",
        reranker=TinyReranker(),
        reranker_id="tiny-reranker-v1",
        default_method="full",
    )
    with pytest.raises(RuntimeError, match="embedding failure"):
        await base.answer("redis", "RDB")
    assert legacy.info("redis")["snapshot_id"] == old["snapshot_id"]


@pytest.mark.asyncio
async def test_discovery_runtime_uses_full_retrieval_for_legacy_imports(
    tmp_path, monkeypatch
):
    from docguard.discovery import runtime

    class Models(TinyEmbedder, TinyReranker):
        embedding_id = "tiny-v1"
        reranker_id = "tiny-reranker-v1"

    monkeypatch.setattr(runtime, "LocalRetrievalModels", lambda *args: Models())
    legacy = make_base(tmp_path / "knowledge")
    await legacy.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    service = runtime.build_discovery_service(ROOT, tmp_path)
    response = await service.query("redis", "save intervals", generate=False)
    assert response["retrieval"]["method"] == "full"
    assert response["evidence"][0]["provenance"]["rerank_score"] == 0.9


@pytest.mark.asyncio
async def test_full_mode_does_not_silently_fall_back_when_models_are_missing(tmp_path):
    base = make_base(tmp_path, default_method="full")
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    with pytest.raises(RuntimeError, match="requires Dense and reranker"):
        await base.answer("redis", "save intervals")


def make_base(root, **kwargs):
    return KnowledgeBase(
        root,
        TokenBudget.from_manifest(ROOT / "data/tokenizers/manifest.json", ROOT),
        load_chunking_config(ROOT / "configs/chunking/s2-chunk-v1.json"),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_url_to_persistent_isolated_search_and_extractive_answer(tmp_path):
    base = make_base(tmp_path)
    first = await base.import_urls(
        "redis",
        [next(iter(PAGES))],
        product="Redis",
        version="7",
        fetcher=FixtureFetcher(),
    )
    assert first["document_count"] == 1
    assert first["chunk_count"] == 2
    repeated = await base.import_urls(
        "redis",
        [next(iter(PAGES))],
        product="Redis",
        version="7",
        fetcher=FixtureFetcher(),
    )
    assert repeated["snapshot_id"] == first["snapshot_id"]
    await base.import_urls(
        "python", [list(PAGES)[1]], product="Python", fetcher=FixtureFetcher()
    )
    reloaded = make_base(tmp_path)
    hits = await reloaded.search("redis", "Redis RDB snapshots")
    assert hits and all(h.chunk.knowledge_base_id == "redis" for h in hits)
    assert all(h.chunk.canonical_url.startswith("https://redis.io/") for h in hits)
    answer = await reloaded.answer("redis", "RDB snapshots")
    assert answer["mode"] == "extractive"
    assert answer["citations"][0]["url"].endswith("#rdb")
    assert (await reloaded.answer("python", "unrelatedxyz"))["action"] == "refuse"


@pytest.mark.asyncio
async def test_failed_import_keeps_previous_snapshot(tmp_path):
    base = make_base(tmp_path)
    good = await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    with pytest.raises((KeyError, ValueError)):
        await base.import_urls(
            "redis",
            ["https://example.com/missing"],
            product="Redis",
            fetcher=FixtureFetcher(),
        )
    assert base.info("redis")["snapshot_id"] == good["snapshot_id"]


@pytest.mark.asyncio
async def test_dense_snapshot_uses_generic_payload_and_isolation(tmp_path):
    base = make_base(tmp_path, embedder=TinyEmbedder(), embedding_id="tiny-fixture-v1")
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    hits = await base.search("redis", "Redis", method="hybrid")
    assert hits and "dense" in hits[0].sources
    assert hits[0].chunk.version is None
    assert hits[0].chunk.product == "Redis"


@pytest.mark.asyncio
async def test_missing_dense_model_is_explicit_not_silent_fallback(tmp_path):
    base = make_base(tmp_path)
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    with pytest.raises(ValueError, match="Dense"):
        await base.search("redis", "Redis", method="hybrid")


@pytest.mark.asyncio
async def test_generated_answer_rejects_unknown_citation(tmp_path):
    from docguard.generation.models import GenerationCompletion

    class Generator:
        async def generate(self, request):
            return GenerationCompletion(
                '{"action":"answer","answer":"Wrong","citations":["E10"]}',
                "fixture",
                "fixture",
                "stop",
                1,
                0,
                1,
            )

    base = make_base(tmp_path)
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    with pytest.raises(ValueError, match="citation"):
        await base.answer("redis", "RDB snapshots", generator=Generator())


@pytest.mark.asyncio
async def test_valid_generated_answer_maps_generic_citations(tmp_path):
    from docguard.generation.models import GenerationCompletion

    class Generator:
        async def generate(self, request):
            assert "postgresql_version" not in request.messages[-1]["content"]
            return GenerationCompletion(
                '{"action":"answer","answer":"RDB stores snapshots.","citations":["E01"]}',
                "fixture",
                "fixture",
                "stop",
                1,
                0,
                1,
            )

    base = make_base(tmp_path)
    await base.import_urls(
        "redis",
        [next(iter(PAGES))],
        product="Redis",
        version="7",
        fetcher=FixtureFetcher(),
    )
    result = await base.answer("redis", "RDB snapshots", generator=Generator())
    assert result["mode"] == "generated"
    assert result["citations"][0]["product"] == "Redis"
    assert result["citations"][0]["version"] == "7"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["refuse", "clarify"])
async def test_generated_non_answer_preserves_action_without_forcing_citations(
    tmp_path, action
):
    import json

    from docguard.generation.models import GenerationCompletion

    class Generator:
        async def generate(self, request):
            return GenerationCompletion(
                json.dumps(
                    {
                        "action": action,
                        "answer": "当前证据没有对象身份与整数缓存的资料。",
                        "citations": [],
                    }
                ),
                "fixture",
                "fixture",
                "stop",
                10,
                0,
                10,
            )

    base = make_base(tmp_path)
    await base.import_urls(
        "python", [list(PAGES)[1]], product="Python", fetcher=FixtureFetcher()
    )
    result = await base.answer(
        "python", "ConfigParser identity and interning", generator=Generator()
    )
    assert result["action"] == result["decision"]["action"] == action
    assert result["answer"] is None
    assert result["explanation"] == "当前证据没有对象身份与整数缓存的资料。"
    assert result["citations"] == []
    assert result["decision"]["confidence"] is None
    assert result["evidence"]


@pytest.mark.asyncio
async def test_negative_statement_in_valid_answer_is_not_a_keyword_refusal(tmp_path):
    from docguard.generation.models import GenerationCompletion

    class Generator:
        async def generate(self, request):
            return GenerationCompletion(
                '{"action":"answer","answer":"This cannot replace RDB snapshots [E01].","citations":["E01"]}',
                "fixture",
                "fixture",
                "stop",
                10,
                0,
                10,
            )

    base = make_base(tmp_path)
    await base.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
    )
    result = await base.answer("redis", "RDB snapshots", generator=Generator())
    assert result["action"] == "answer"
    assert result["answer"].startswith("This cannot")


@pytest.mark.parametrize(
    "payload",
    [
        {"answer": "text", "citations": []},
        {"action": "unknown", "answer": "text", "citations": []},
        {"action": "refuse", "answer": "text", "citations": ["E01"]},
    ],
)
def test_generic_decision_contract_rejects_missing_unknown_or_conflicting_actions(
    payload,
):
    import json

    from docguard.knowledge.pipeline import _parse_knowledge_output

    with pytest.raises(ValueError):
        _parse_knowledge_output(json.dumps(payload))


@pytest.mark.asyncio
async def test_source_hash_tampering_is_rejected(tmp_path):
    class TamperedFetcher(FixtureFetcher):
        def fetch(self, url):
            from dataclasses import replace

            return replace(super().fetch(url), content_sha256="0" * 64)

    with pytest.raises(ValueError, match="hash"):
        await make_base(tmp_path).import_urls(
            "redis", [next(iter(PAGES))], product="Redis", fetcher=TamperedFetcher()
        )


@pytest.mark.asyncio
async def test_failed_dense_build_never_publishes(tmp_path):
    class BrokenEmbedder(TinyEmbedder):
        async def embed_passages(self, texts):
            return [[float("nan"), 1.0, 0.0] for _ in texts]

    base = make_base(tmp_path, embedder=BrokenEmbedder(), embedding_id="broken")
    with pytest.raises(ValueError, match="finite"):
        await base.import_urls(
            "redis", [next(iter(PAGES))], product="Redis", fetcher=FixtureFetcher()
        )
    assert not (tmp_path / "redis/active.json").exists()


@pytest.mark.parametrize("name", ["../outside", "a/b", "", "A B"])
def test_knowledge_base_id_cannot_escape_storage(tmp_path, name):
    with pytest.raises(ValueError):
        make_base(tmp_path).info(name)


@pytest.mark.asyncio
async def test_import_reports_real_phase_and_embedding_batch_counts(tmp_path):
    from dataclasses import replace

    class ManyPages(FixtureFetcher):
        def fetch(self, url):
            snapshot = super().fetch(url)
            body = (
                "<main>"
                + "".join(
                    f'<h2 id="s{i}">Section {i}</h2><p>Redis persists data safely.</p>'
                    for i in range(40)
                )
                + "</main>"
            ).encode()
            return replace(
                snapshot, body=body, content_sha256=hashlib.sha256(body).hexdigest()
            )

    class CountingEmbedder(TinyEmbedder):
        def __init__(self):
            self.batches = []

        async def embed_passages(self, texts):
            self.batches.append(len(texts))
            return await super().embed_passages(texts)

    embedder = CountingEmbedder()
    base = make_base(
        tmp_path / "with-progress", embedder=embedder, embedding_id="tiny-v1"
    )
    progress = []
    info = await base.import_urls(
        "redis",
        [next(iter(PAGES))],
        product="Redis",
        fetcher=ManyPages(),
        progress=progress.append,
    )
    assert info["chunk_count"] == 40
    assert embedder.batches == [32, 8]
    assert {
        "parsing",
        "chunking",
        "bm25",
        "embedding",
        "dense_index",
        "publishing",
    } <= {event["phase"] for event in progress}
    assert [
        (event["completed"], event["total"])
        for event in progress
        if event["phase"] == "embedding"
    ] == [(0, 40), (32, 40), (40, 40)]
    plain = make_base(
        tmp_path / "plain", embedder=TinyEmbedder(), embedding_id="tiny-v1"
    )
    control = await plain.import_urls(
        "redis", [next(iter(PAGES))], product="Redis", fetcher=ManyPages()
    )
    assert control["snapshot_id"] == info["snapshot_id"]

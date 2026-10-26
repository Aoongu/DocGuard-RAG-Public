"""Reproduce local Markdown ingestion and semantic search with public Ragbits APIs."""

import asyncio
import json
from pathlib import Path

from ragbits.core.embeddings.dense.fastembed import FastEmbedEmbedder
from ragbits.core.vector_stores.in_memory import InMemoryVectorStore
from ragbits.document_search import DocumentSearch
from ragbits.document_search.documents.document import DocumentMeta

ROOT = Path(__file__).resolve().parent
INPUTS_DIR = ROOT / "inputs"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "document_search.json"

QUERIES = (
    ("Which Python toolkit coordinates ingestion and semantic retrieval?", "ragbits_overview.md"),
    ("What stores embeddings locally without a database server?", "vector_store.md"),
    ("Which metric checks whether the relevant document ranks first?", "evaluation.md"),
)


async def run_reproduction(output_path: Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, str | float]]:
    """Ingest the fixture documents, run fixed queries, and persist their top results."""
    documents = [DocumentMeta.from_local_path(INPUTS_DIR / filename) for _, filename in QUERIES]
    embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    document_search = DocumentSearch(vector_store=InMemoryVectorStore(embedder=embedder))
    await document_search.ingest(documents)

    records: list[dict[str, str | float]] = []
    for query, _ in QUERIES:
        top_result = (await document_search.search(query))[0]
        records.append(
            {
                "query": query,
                "source": Path(top_result.document_meta.source.path).name,
                "score": float(top_result.score),
                "text": top_result.text_representation,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records


if __name__ == "__main__":
    asyncio.run(run_reproduction())

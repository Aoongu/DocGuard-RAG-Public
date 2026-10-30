from types import SimpleNamespace


def test_postgresql_api_keeps_generation_labels_separate_from_chunk_identifiers():
    from docguard.service.query import _citation

    value = SimpleNamespace(
        evidence_id="E03",
        chunk_id="chunk-third",
        postgresql_version="18.6",
        topic="vacuum",
        section_path=("VACUUM",),
        canonical_url="https://www.postgresql.org/docs/18/sql-vacuum.html",
        anchor="VACUUM",
        text="source",
        score=0.8,
    )
    citation = _citation(value)
    assert citation.evidence_id == "chunk-third"
    assert citation.citation_label == "E03"

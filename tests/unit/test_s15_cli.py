import pytest

from docguard.knowledge.cli import main


def test_info_missing_database_is_sanitized(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "info", "--kb", "redis"]) == 2
    output = capsys.readouterr()
    assert "knowledge_error" in output.err
    assert str(tmp_path) not in output.err


def test_import_requires_explicit_urls():
    with pytest.raises(SystemExit) as error:
        main(["import", "--kb", "redis", "--product", "Redis"])
    assert error.value.code == 2

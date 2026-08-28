import os
import stat

import pytest

from docling_graph_service.pipeline import _ResultCache
from docling_graph_service.schemas import Degraded, DocumentInfo, ProcessResponse


def _resp():
    return ProcessResponse(document=DocumentInfo(name="x.pdf", format="pdf", sha256="0" * 64, pages=1, tables=0, pictures=0),
                           markdown="# x", chunks=[], graph=None, degraded=Degraded())


def test_roundtrip(tmp_path):
    cache = _ResultCache(str(tmp_path))
    cache.save_result("k", _resp())
    assert cache.load_result("k").markdown == "# x"
    assert cache.load_result("missing") is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_cache_never_raises(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        cache = _ResultCache(str(ro))
        cache.save_result("k", _resp())  # PermissionError swallowed (logged)
        assert cache.load_result("k") is None
        assert _ResultCache(str(ro / "sub" / "dir")).dir is None  # cannot create -> disabled
    finally:
        ro.chmod(stat.S_IRWXU)

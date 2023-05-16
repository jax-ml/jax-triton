import pathlib
import pytest

import mktestdocs

# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize('fpath', [pathlib.Path("docs") / "guides" /
                                   "pallas.md"], ids=str)
def test_guides(fpath):
  mktestdocs.check_md_file(fpath=fpath, memory=True)

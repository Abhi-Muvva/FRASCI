from pathlib import Path

import pytest

from frasci import FCIDUMPInput


def test_explicit_input_validates_counts(tmp_path):
    path = tmp_path / "FCIDUMP"
    path.write_text("placeholder")
    value = FCIDUMPInput(path, n_orb=2, n_alpha=1, n_beta=1)
    assert value.fcidump_path == path
    with pytest.raises(ValueError, match="cannot exceed"):
        FCIDUMPInput(path, n_orb=2, n_alpha=3, n_beta=0)


def test_from_file_reads_small_physical_fcidump(tmp_path):
    path = tmp_path / "FCIDUMP"
    path.write_text(
        "&FCI NORB=2, NELEC=2, MS2=0,\n"
        " ORBSYM=1,1,\n"
        " ISYM=1,\n"
        "&END\n"
        " 0.7000000000000000  1  1  1  1\n"
        " 0.6000000000000000  2  2  2  2\n"
        " 0.2000000000000000  1  1  2  2\n"
        "-1.0000000000000000  1  1  0  0\n"
        "-0.5000000000000000  2  2  0  0\n"
        " 0.0000000000000000  0  0  0  0\n"
    )
    value = FCIDUMPInput.from_file(path)
    assert (value.n_orb, value.n_alpha, value.n_beta) == (2, 1, 1)

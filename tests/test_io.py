import numpy as np
import pytest

import morphanalyzer as ma


def test_raw_roundtrip(tmp_path, pack):
    p = tmp_path / "vol.raw"
    ma.io.write_raw(pack, p, dtype="uint8")
    assert p.with_suffix(".raw.json").exists()
    back = ma.io.read_raw(p, shape=pack.shape, dtype="uint8", mmap=False)
    assert np.array_equal(back.data.astype(bool), pack.solid)


def test_raw_detects_wrong_shape(tmp_path, pack):
    p = tmp_path / "vol.raw"
    ma.io.write_raw(pack, p, dtype="uint8")
    with pytest.raises(ValueError, match="verifier shape/dtype"):
        ma.io.read_raw(p, shape=(999, 999, 999), dtype="uint8")


def test_stack_roundtrip_multipage(tmp_path, pack):
    p = tmp_path / "vol.tif"
    ma.io.write_stack(pack, p)
    back = ma.io.read_stack(p, voxel_size=pack.voxel_size[0])
    assert back.shape == pack.shape
    assert np.array_equal(back.data > 0, pack.solid)


def test_stack_roundtrip_per_slice_natural_order(tmp_path):
    vol = ma.phantoms.sphere(shape=(12, 16, 16), radius=5.0)
    d = tmp_path / "slices"
    ma.io.write_stack(vol, d, per_slice=True)
    assert len(list(d.glob("*.tif"))) == 12
    back = ma.io.read_stack(d)
    assert np.array_equal(back.data > 0, vol.solid)


def test_table_roundtrip(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"label": [1, 2], "volume": [10.5, 20.25]})
    p = ma.io.write_table(df, tmp_path / "cells.csv")
    back = ma.io.read_table(p)
    pd.testing.assert_frame_equal(df, back)

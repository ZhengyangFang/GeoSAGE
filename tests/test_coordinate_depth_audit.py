from __future__ import annotations

import numpy as np
import discretize

from geosage.evidence_bundle import build_evidence_bundle


def test_target_depth_audit_uses_local_topography(tmp_path) -> None:
    """Target depth must be derived from local surface elevation minus model z."""

    topo_path = tmp_path / "topography.xyz"
    np.savetxt(
        topo_path,
        np.array(
            [
                [0.5, 0.5, 10.0],
                [1.5, 0.5, 11.0],
                [0.5, 1.5, 12.0],
                [1.5, 1.5, 13.0],
            ]
        ),
    )
    mesh = discretize.TensorMesh([np.ones(2), np.ones(2), np.ones(2)])
    geo_ids = np.zeros((2, 2, 2), dtype=int)
    geo_ids[:, :, 0] = 5

    bundle = build_evidence_bundle(
        {
            "source_manifest": {"artifacts": {"topography": {"path": str(topo_path)}}},
            "inversion_result": {
                "mesh_core": mesh,
                "dens_core_3d": np.zeros((2, 2, 2)),
                "susc_core_3d": np.zeros((2, 2, 2)),
            },
            "geology_result": {"mesh_core": mesh, "geo_id_3d": geo_ids, "geo_defs": {5: "Target"}},
        },
        {},
        {},
        target_geo_ids=[5],
    )

    reference = bundle["coordinate_reference"]
    record = bundle["target_depth_audit"]["records"][0]
    assert reference["topography"]["available"] is True
    assert reference["k_0_z_elevation_m"] == 0.5
    assert reference["k_last_z_elevation_m"] == 1.5
    assert record["depth_below_local_surface_m"]["p50"] == 11.0
    assert record["depth_band_fraction"]["0-1000 m"] == 1.0

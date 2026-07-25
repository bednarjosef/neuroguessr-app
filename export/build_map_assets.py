#!/usr/bin/env python
"""
Build a compact, fully-offline world basemap binary from Natural Earth vector data.

Input : data/naturalearth/*.geojson  (Natural Earth 1:10m + 1:50m + 1:110m, public domain)
Output: mobile/assets_build/map/world.bin
        mobile/assets_build/map/names.txt
        mobile/assets_build/map/world_meta.json

Layers (format v2): countries_110m, countries_50m, coastline_50m and lakes_50m carry
polygon/line geometry; places_10m and country_labels are POINT layers whose records
live in the PLACE TABLE, carry a SCALERANK-derived importance rank plus POP_MAX and a
capital flag, and are sorted by rank so the renderer can draw a prefix per zoom level.

The binary is a flat, mmap-friendly container -- see FORMAT.md / the module docstring of
verify_map.py for the byte-level spec. Nothing in it needs a JSON parser at runtime.

Usage:
    .venv/bin/python mobile/export/build_map_assets.py
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

REPO = Path(__file__).resolve().parents[2]
NE_DIR = REPO / "data" / "naturalearth"
OUT_DIR = REPO / "mobile" / "assets_build" / "map"

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"

MAGIC = 0x4D57474E  # bytes 'N','G','W','M' little-endian
VERSION = 2
COORD_SCALE = 1_000_000  # degrees = int32 / 1e6  -> ~0.11 m precision

GEOM_POLYGON = 0
GEOM_LINE = 1
GEOM_POINT = 2

HEADER_SIZE = 128  # v2 header (v1 was 64; the first 64 bytes are unchanged)
LAYER_REC = 16
FEATURE_REC = 28
RING_REC = 8
PLACE_REC = 24
RANK_SLOTS = 16  # rank index entries per point layer; ranks are clamped to 0..14

# layer spec: (layer name, geojson file, geom type, simplify tolerance in deg,
#              min part area in sq deg, name property or None)
LAYERS = [
    ("countries_110m", "ne_110m_admin_0_countries.geojson", GEOM_POLYGON, 0.05, 0.01, "ADMIN"),
    ("countries_50m", "ne_50m_admin_0_countries.geojson", GEOM_POLYGON, 0.001, 0.00005, "ADMIN"),
    ("coastline_50m", "ne_50m_coastline.geojson", GEOM_LINE, 0.001, 0.0, None),
    ("lakes_50m", "ne_50m_lakes.geojson", GEOM_POLYGON, 0.002, 0.0005, "name"),
]

PLACES_FILE = "ne_10m_populated_places.geojson"
COUNTRIES_FILE = "ne_50m_admin_0_countries.geojson"
PLACES_LAYER = "places_10m"
COUNTRY_LABEL_LAYER = "country_labels"

# keep every populated place with SCALERANK <= this (10m has ranks 0..10)
PLACE_MAX_RANK = 14

# place flags (bit -> meaning)
FLAG_CAPITAL = 0x01      # national (admin-0) capital
FLAG_ADM1_CAPITAL = 0x02  # state / province / region capital
FLAG_MEGACITY = 0x04
FLAG_WORLDCITY = 0x08
FLAG_COUNTRY_LABEL = 0x10  # record is a country label anchor, not a settlement

EXTRA_FILES = [PLACES_FILE]

PLACE_DTYPE = np.dtype([
    ("name", "<u4"), ("lonQ", "<i4"), ("latQ", "<i4"), ("popMax", "<u4"),
    ("country", "<u4"), ("rank", "u1"), ("flags", "u1"), ("labelRank", "u1"),
    ("reserved", "u1"),
])


# --------------------------------------------------------------------------- utils
def ensure_downloaded() -> None:
    NE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [spec[1] for spec in LAYERS] + EXTRA_FILES
    for fname in wanted:
        dst = NE_DIR / fname
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        url = f"{NE_BASE}/{fname}"
        print(f"[fetch] {url}")
        urllib.request.urlretrieve(url, dst)


def iter_parts(geom: BaseGeometry, geom_type: int):
    """Yield (rings) per geometry part. For polygons: [exterior, hole, hole...].
    For lines: [linestring] (one ring per part)."""
    gt = geom.geom_type
    if gt == "Polygon":
        yield [list(geom.exterior.coords)] + [list(r.coords) for r in geom.interiors]
    elif gt == "MultiPolygon":
        for p in geom.geoms:
            yield from iter_parts(p, geom_type)
    elif gt == "LineString":
        yield [list(geom.coords)]
    elif gt == "MultiLineString":
        for ls in geom.geoms:
            yield [list(ls.coords)]
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            yield from iter_parts(g, geom_type)
    # anything else (Point etc.) is skipped


def quantize(ring, closed: bool):
    """Round to the int32 grid and drop consecutive duplicates."""
    a = np.asarray(ring, dtype=np.float64)[:, :2]
    q = np.rint(a * COORD_SCALE).astype(np.int64)
    np.clip(q, -180 * COORD_SCALE, 180 * COORD_SCALE, out=q)
    if len(q) > 1:
        keep = np.ones(len(q), dtype=bool)
        keep[1:] = np.any(q[1:] != q[:-1], axis=1)
        q = q[keep]
    if closed and len(q) >= 2 and np.array_equal(q[0], q[-1]):
        # store rings WITHOUT the repeated closing vertex; the renderer closes them
        q = q[:-1]
    return q.astype(np.int32)


# --------------------------------------------------------------------------- build
class Builder:
    def __init__(self) -> None:
        self.names: list[str] = [""]
        self.name_idx: dict[str, int] = {"": 0}
        self.layers: list[tuple[int, int, int, int]] = []  # nameId, firstRecord, nRecords, geomType
        self.features: list[tuple[int, int, int, float, float, float, float]] = []
        self.rings: list[tuple[int, int]] = []
        self.coords: list[np.ndarray] = []
        self.n_points = 0
        # place table: (nameId, lonQ, latQ, popMax, countryNameId, rank, flags, labelRank)
        self.places: list[tuple[int, int, int, int, int, int, int, int]] = []
        self.rank_index: list[np.ndarray] = []  # one 16-entry prefix table per point layer

    def name_id(self, s: str | None) -> int:
        s = (s or "").strip()
        if s not in self.name_idx:
            self.name_idx[s] = len(self.names)
            self.names.append(s)
        return self.name_idx[s]

    def add_part(self, name_id: int, rings_q: list[np.ndarray]) -> None:
        first_ring = len(self.rings)
        mnx = mny = 1e18
        mxx = mxy = -1e18
        for rq in rings_q:
            self.rings.append((self.n_points, len(rq)))
            self.coords.append(rq)
            self.n_points += len(rq)
            mnx = min(mnx, rq[:, 0].min())
            mxx = max(mxx, rq[:, 0].max())
            mny = min(mny, rq[:, 1].min())
            mxy = max(mxy, rq[:, 1].max())
        self.features.append(
            (
                name_id,
                first_ring,
                len(rings_q),
                mnx / COORD_SCALE,
                mny / COORD_SCALE,
                mxx / COORD_SCALE,
                mxy / COORD_SCALE,
            )
        )

    def add_layer(self, layer_name, fname, geom_type, tol, min_area, name_prop):
        path = NE_DIR / fname
        gj = json.loads(path.read_text())
        first_feat = len(self.features)
        ring0, pt0 = len(self.rings), self.n_points
        n_parts = 0
        for feat in gj["features"]:
            geom = feat.get("geometry")
            if not geom:
                continue
            g = shape(geom)
            if g.is_empty:
                continue
            if tol > 0:
                g = g.simplify(tol, preserve_topology=True)
                if g.is_empty:
                    continue
            props = feat.get("properties", {}) or {}
            nid = self.name_id(props.get(name_prop) if name_prop else "")

            parts = list(iter_parts(g, geom_type))
            if geom_type == GEOM_POLYGON and min_area > 0 and len(parts) > 1:
                # keep parts above the area floor, but never drop the largest part
                areas = [_ring_area(p[0]) for p in parts]
                biggest = int(np.argmax(areas))
                parts = [p for i, p in enumerate(parts) if areas[i] >= min_area or i == biggest]

            closed = geom_type == GEOM_POLYGON
            min_pts = 3 if closed else 2
            for rings in parts:
                rq = []
                for i, r in enumerate(rings):
                    q = quantize(r, closed)
                    if len(q) < min_pts:
                        if i == 0:
                            rq = []
                            break
                        continue
                    rq.append(q)
                if rq:
                    self.add_part(nid, rq)
                    n_parts += 1
        self.layers.append((self.name_id(layer_name), first_feat, n_parts, geom_type))
        print(
            f"[layer] {layer_name:<16} features={n_parts:>6} "
            f"rings={len(self.rings)-ring0:>6} points={self.n_points-pt0:>8} "
            f"({(self.n_points-pt0)*8/1e6:.2f} MB of coords)"
        )

    # ---- point layers ------------------------------------------------------
    def _add_point_layer(self, layer_name: str, recs: list[tuple]) -> None:
        """recs = (rank, popMax, name, countryName, lonDeg, latDeg, flags, labelRank).
        Records are sorted by (rank asc, popMax desc, name) so a renderer can draw a
        prefix of the layer and get the most important places first."""
        recs = sorted(recs, key=lambda r: (r[0], -r[1], r[2]))
        first = len(self.places)
        counts = np.zeros(RANK_SLOTS + 1, dtype=np.int64)
        for rank, pop, name, cname, lon, lat, flags, lrank in recs:
            lonq = int(np.clip(round(lon * COORD_SCALE), -180 * COORD_SCALE, 180 * COORD_SCALE))
            latq = int(np.clip(round(lat * COORD_SCALE), -90 * COORD_SCALE, 90 * COORD_SCALE))
            self.places.append((self.name_id(name), lonq, latq, int(min(max(pop, 0), 0xFFFFFFFF)),
                                self.name_id(cname), rank, flags, lrank))
            counts[rank] += 1
        # rankStart[r] = number of records in this layer with rank strictly < r
        prefix = np.zeros(RANK_SLOTS, dtype=np.uint32)
        acc = 0
        for r in range(RANK_SLOTS):
            prefix[r] = acc
            acc += int(counts[r])
        self.rank_index.append(prefix)
        self.layers.append((self.name_id(layer_name), first, len(recs), GEOM_POINT))
        hist = " ".join(f"r{r}={int(counts[r])}" for r in range(RANK_SLOTS) if counts[r])
        print(f"[layer] {layer_name:<16} points={len(recs):>6} "
              f"({len(recs)*PLACE_REC/1e3:.1f} kB of records)  {hist}")

    def add_places(self, fname: str, layer_name: str = PLACES_LAYER,
                   max_rank: int = PLACE_MAX_RANK) -> None:
        gj = json.loads((NE_DIR / fname).read_text())
        recs, dropped = [], 0
        for feat in gj["features"]:
            p = feat.get("properties", {}) or {}
            name = (p.get("NAME") or p.get("NAMEASCII") or "").strip()
            geom = feat.get("geometry") or {}
            if not name or geom.get("type") != "Point":
                dropped += 1
                continue
            rank = p.get("SCALERANK")
            rank = RANK_SLOTS - 2 if rank is None else int(rank)
            rank = max(0, min(rank, RANK_SLOTS - 2))
            if rank > max_rank:
                dropped += 1
                continue
            # LATITUDE/LONGITUDE props are the authoritative NE values; fall back to geometry
            lon = float(p["LONGITUDE"]) if p.get("LONGITUDE") is not None else float(geom["coordinates"][0])
            lat = float(p["LATITUDE"]) if p.get("LATITUDE") is not None else float(geom["coordinates"][1])
            fcla = (p.get("FEATURECLA") or "")
            flags = 0
            if int(p.get("ADM0CAP") or 0) == 1 or fcla.startswith("Admin-0 capital"):
                flags |= FLAG_CAPITAL
            if fcla.startswith("Admin-1"):
                flags |= FLAG_ADM1_CAPITAL
            if int(p.get("MEGACITY") or 0) == 1:
                flags |= FLAG_MEGACITY
            if int(p.get("WORLDCITY") or 0) == 1:
                flags |= FLAG_WORLDCITY
            lrank = p.get("LABELRANK")
            lrank = 255 if lrank is None else max(0, min(int(lrank), 255))
            recs.append((rank, int(p.get("POP_MAX") or 0), name,
                         (p.get("ADM0NAME") or "").strip(), lon, lat, flags, lrank))
        self._add_point_layer(layer_name, recs)
        if dropped:
            print(f"         (dropped {dropped} unnamed / non-point / rank>{max_rank} records)")

    def add_country_labels(self, fname: str = COUNTRIES_FILE,
                           layer_name: str = COUNTRY_LABEL_LAYER) -> None:
        """One label anchor per country. Prefer Natural Earth's curated LABEL_X/LABEL_Y,
        but only when it actually falls inside the country; otherwise use
        representative_point() of the LARGEST polygon part (never the centroid, which
        can land in the sea for e.g. Indonesia or Norway)."""
        gj = json.loads((NE_DIR / fname).read_text())
        recs, n_ne, n_rep = [], 0, 0
        for feat in gj["features"]:
            geom = feat.get("geometry")
            p = feat.get("properties", {}) or {}
            name = (p.get("ADMIN") or p.get("NAME") or "").strip()
            if not geom or not name:
                continue
            g = shape(geom)
            if g.is_empty:
                continue
            parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            biggest = max(parts, key=lambda q: q.area)
            lx, ly = p.get("LABEL_X"), p.get("LABEL_Y")
            pt = None
            if lx is not None and ly is not None:
                from shapely.geometry import Point
                cand = Point(float(lx), float(ly))
                if g.contains(cand):
                    pt, n_ne = cand, n_ne + 1
            if pt is None:
                pt, n_rep = biggest.representative_point(), n_rep + 1
            rank = p.get("LABELRANK")
            rank = RANK_SLOTS - 2 if rank is None else int(rank)
            rank = max(0, min(rank, RANK_SLOTS - 2))
            recs.append((rank, int(p.get("POP_EST") or 0), name, name,
                         float(pt.x), float(pt.y), FLAG_COUNTRY_LABEL, rank))
        self._add_point_layer(layer_name, recs)
        print(f"         (anchor source: {n_ne} NE LABEL_X/Y inside polygon, "
              f"{n_rep} shapely representative_point of largest part)")

    def write(self, out_path: Path) -> None:
        name_blob = b"".join(s.encode("utf-8") for s in self.names)
        name_offsets = np.zeros(len(self.names) + 1, dtype=np.uint32)
        acc = 0
        for i, s in enumerate(self.names):
            acc += len(s.encode("utf-8"))
            name_offsets[i + 1] = acc

        n_point_layers = sum(1 for l in self.layers if l[3] == GEOM_POINT)
        layer_off = HEADER_SIZE
        feature_off = layer_off + LAYER_REC * len(self.layers)
        ring_off = feature_off + FEATURE_REC * len(self.features)
        coord_off = ring_off + RING_REC * len(self.rings)
        place_off = coord_off + 8 * self.n_points
        rank_off = place_off + PLACE_REC * len(self.places)
        name_tab_off = rank_off + 4 * RANK_SLOTS * n_point_layers
        name_blob_off = name_tab_off + 4 * (len(self.names) + 1)
        file_bytes = name_blob_off + len(name_blob)

        hdr = struct.pack(
            "<32I",
            MAGIC,
            VERSION,
            len(self.layers),
            len(self.features),
            len(self.rings),
            self.n_points,
            len(self.names),
            COORD_SCALE,
            layer_off,
            feature_off,
            ring_off,
            coord_off,
            name_tab_off,
            name_blob_off,
            len(name_blob),
            HEADER_SIZE,        # [15] headerBytes (v1 wrote 0 here, meaning 64)
            n_point_layers,     # [16]
            len(self.places),   # [17]
            place_off,          # [18]
            PLACE_REC,          # [19]
            rank_off,           # [20]
            RANK_SLOTS,         # [21]
            LAYER_REC,          # [22]
            FEATURE_REC,        # [23]
            RING_REC,           # [24]
            8,                  # [25] coordRecBytes
            file_bytes,         # [26]
            0, 0, 0, 0, 0,      # [27..31] reserved
        )
        assert len(hdr) == HEADER_SIZE

        layer_arr = np.array(self.layers, dtype=np.uint32)

        feat = np.zeros(len(self.features), dtype=np.dtype(
            [("name", "<u4"), ("first", "<u4"), ("n", "<u4"),
             ("mnx", "<f4"), ("mny", "<f4"), ("mxx", "<f4"), ("mxy", "<f4")]))
        feat["name"] = [f[0] for f in self.features]
        feat["first"] = [f[1] for f in self.features]
        feat["n"] = [f[2] for f in self.features]
        feat["mnx"] = np.float32([f[3] for f in self.features])
        feat["mny"] = np.float32([f[4] for f in self.features])
        feat["mxx"] = np.float32([f[5] for f in self.features])
        feat["mxy"] = np.float32([f[6] for f in self.features])
        # widen float32 bboxes outward so they never clip real geometry
        eps = np.float32(2e-4)
        feat["mnx"] = (feat["mnx"] - eps).astype("<f4")
        feat["mny"] = (feat["mny"] - eps).astype("<f4")
        feat["mxx"] = (feat["mxx"] + eps).astype("<f4")
        feat["mxy"] = (feat["mxy"] + eps).astype("<f4")
        assert feat.itemsize == FEATURE_REC

        ring_arr = np.array(self.rings, dtype="<u4")
        coord_arr = np.concatenate(self.coords).astype("<i4") if self.coords else np.zeros((0, 2), "<i4")

        place = np.zeros(len(self.places), dtype=PLACE_DTYPE)
        for i, (nid, lonq, latq, pop, cnid, rank, flags, lrank) in enumerate(self.places):
            place[i] = (nid, lonq, latq, pop, cnid, rank, flags, lrank, 0)
        assert place.itemsize == PLACE_REC
        rank_arr = (np.concatenate(self.rank_index) if self.rank_index
                    else np.zeros(0, dtype=np.uint32)).astype("<u4")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(hdr)
            f.write(layer_arr.astype("<u4").tobytes())
            f.write(feat.tobytes())
            f.write(ring_arr.tobytes())
            f.write(coord_arr.tobytes())
            f.write(place.tobytes())
            f.write(rank_arr.tobytes())
            f.write(name_offsets.astype("<u4").tobytes())
            f.write(name_blob)
        assert out_path.stat().st_size == file_bytes, "header fileBytes mismatch"
        print(f"[write] {out_path}  {out_path.stat().st_size/1e6:.2f} MB")

        (out_path.parent / "names.txt").write_text("\n".join(self.names) + "\n", encoding="utf-8")
        meta = {
            "magic": "NGWM",
            "version": VERSION,
            "coord_scale": COORD_SCALE,
            "bytes": out_path.stat().st_size,
            "header_bytes": HEADER_SIZE,
            "layers": [
                {"name": self.names[l[0]], "first_record": int(l[1]),
                 "record_count": int(l[2]),
                 "geom_type": {0: "polygon", 1: "line", 2: "point"}[int(l[3])]}
                for l in self.layers
            ],
            "feature_count": len(self.features),
            "ring_count": len(self.rings),
            "point_count": self.n_points,
            "place_count": len(self.places),
            "place_rec_bytes": PLACE_REC,
            "rank_slots": RANK_SLOTS,
            "name_count": len(self.names),
            "flags": {"capital": FLAG_CAPITAL, "adm1_capital": FLAG_ADM1_CAPITAL,
                      "megacity": FLAG_MEGACITY, "worldcity": FLAG_WORLDCITY,
                      "country_label": FLAG_COUNTRY_LABEL},
            "source": "Natural Earth 1:10m + 1:50m + 1:110m (public domain)",
        }
        (out_path.parent / "world_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _ring_area(ring) -> float:
    a = np.asarray(ring, dtype=np.float64)[:, :2]
    x, y = a[:, 0], a[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "world.bin"))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    if not args.skip_download:
        ensure_downloaded()

    b = Builder()
    for spec in LAYERS:
        b.add_layer(*spec)
    b.add_places(PLACES_FILE)
    b.add_country_labels()
    b.write(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Decoder / validator for the offline basemap binary produced by build_map_assets.py.

Run:  .venv/bin/python mobile/export/verify_map.py
      .venv/bin/python mobile/export/verify_map.py --compare old_v1.bin

It (1) parses mobile/assets_build/map/world.bin straight out of an mmap, (2) prints
feature/ring/point/place counts and the file size, (3) renders a world preview PNG
with place labels, (4) point-in-polygon tests known coordinates against the
countries_50m layer, (5) validates the point (place) layers, and with --compare
(6) proves the polygon/line layers decode bit-identically against an older file.

================================================================================
FILE FORMAT SPEC -- world.bin  (magic "NGWM", version 2)
All integers and floats are LITTLE-ENDIAN. All sections are 4-byte aligned.
Every *Offset field is an absolute byte offset from the start of the file.

v2 vs v1: the header grew 64 -> 128 bytes, the rivers_50m layer is gone, and two
POINT layers were added (places_10m, country_labels) backed by a new PLACE TABLE
plus a RANK INDEX. Header fields 0..56 and the LAYER / FEATURE / RING / COORD /
NAME sections are byte-for-byte the same shape as v1, so a v1 reader that follows
the absolute offsets still decodes every polygon and line layer correctly; it only
has to skip layers whose geomType is > 1.
================================================================================

HEADER -- 128 bytes at offset 0, thirty-two u32 fields:
  off  type  field
    0  u32   magic          = 0x4D57474E  (ASCII bytes 'N','G','W','M' in file order)
    4  u32   version        = 2
    8  u32   layerCount             ALL layers, polygon + line + point
   12  u32   featureCount           records in the FEATURE TABLE (polygon/line only)
   16  u32   ringCount
   20  u32   pointCount             VERTEX count in the coordinate blob (not places!)
   24  u32   nameCount
   28  u32   coordScale     = 1000000   (degrees = int32Value / coordScale)
   32  u32   layerTableOffset
   36  u32   featureTableOffset
   40  u32   ringTableOffset
   44  u32   coordOffset
   48  u32   nameOffsetTableOffset
   52  u32   nameBlobOffset
   56  u32   nameBlobBytes
   60  u32   headerBytes    = 128       (v1 files carry 0 here, which means 64)
   64  u32   pointLayerCount        number of layers with geomType == 2
   68  u32   placeCount             records in the PLACE TABLE (all point layers)
   72  u32   placeTableOffset
   76  u32   placeRecBytes  = 24        stride of a place record
   80  u32   rankIndexOffset
   84  u32   rankIndexEntries = 16      u32 entries per point layer in the rank index
   88  u32   layerRecBytes  = 16
   92  u32   featureRecBytes = 28
   96  u32   ringRecBytes   = 8
  100  u32   coordRecBytes  = 8
  104  u32   fileBytes                  total size of world.bin, for a cheap check
  108..124  u32 reserved[5] = 0
  The *RecBytes fields exist so a reader can stride by the value in the file rather
  than by a hard-coded constant if the format is ever widened again.

LAYER TABLE -- layerCount records of 16 bytes at layerTableOffset:
    0  u32   nameId          index into the name table -> layer name string
    4  u32   firstRecord     index of this layer's first record
    8  u32   recordCount     number of consecutive records owned by this layer
   12  u32   geomType        0 = POLYGON, 1 = LINE, 2 = POINT
  geomType 0/1 -> firstRecord/recordCount index the FEATURE TABLE.
  geomType 2   -> firstRecord/recordCount index the PLACE TABLE.
  Records are stored grouped by layer, so layer i owns indices
  [firstRecord, firstRecord + recordCount) of its own table.
  A v1-era reader must ignore layers with geomType > 1; their firstRecord is NOT a
  feature index.  Layers in this build, in file order:
    0 countries_110m  POLYGON
    1 countries_50m   POLYGON
    2 coastline_50m   LINE
    3 lakes_50m       POLYGON
    4 places_10m      POINT
    5 country_labels  POINT

FEATURE TABLE -- featureCount records of 28 bytes at featureTableOffset:
    0  u32   nameId          index into the name table (0 == empty string)
    4  u32   firstRing       index of this feature's first record in the ring table
    8  u32   ringCount       number of consecutive ring records
   12  f32   minLon          bounding box, DEGREES (already decoded; for cheap culling)
   16  f32   minLat
   20  f32   maxLon
   24  f32   maxLat
  POLYGON layers: one feature == ONE polygon part. ring[0] is the exterior ring,
  ring[1..ringCount-1] are holes. A country with N islands appears as N features
  that all share the same nameId -- group by nameId to get the whole country.
  LINE layers: one feature == one polyline part; ringCount is normally 1 and
  there is no exterior/hole distinction.
  The bboxes are float32 rounded OUTWARD by 2e-4 deg, so they never clip geometry.

RING TABLE -- ringCount records of 8 bytes at ringTableOffset:
    0  u32   firstPoint      index of the first point IN POINTS (not bytes)
    4  u32   pointCount      number of points in this ring
  Byte address of the ring's coordinates = coordOffset + firstPoint * 8.

COORDINATE BLOB -- pointCount records of 8 bytes at coordOffset:
    0  i32   lonQ            lonDegrees = lonQ / coordScale
    4  i32   latQ            latDegrees = latQ / coordScale
  coordScale = 1e6 -> ~0.11 m quantization step. Points are in ring order.
  POLYGON rings are stored WITHOUT the repeated closing vertex: the last point
  is NOT equal to the first, so the renderer/PIP must close the ring itself
  (Path.close() / lineTo(first)). Rings have >= 3 points (polygons) or >= 2 (lines).
  Ring winding is whatever Natural Earth used -- do NOT rely on it; use ring
  index 0 == exterior instead.

PLACE TABLE -- placeCount records of 24 bytes at placeTableOffset:
    0  u32   nameId          index into the name table -> the place name (UTF-8)
    4  i32   lonQ            lonDegrees = lonQ / coordScale
    8  i32   latQ            latDegrees = latQ / coordScale
   12  u32   popMax          Natural Earth POP_MAX (metro population estimate;
                             0 when unknown; for country_labels it is POP_EST)
   16  u32   countryNameId   index into the name table -> country the place is in
                             (NE ADM0NAME; for country_labels == nameId)
   20  u8    rank            importance, 0 = most important. See RANK below.
   21  u8    flags           bit0 0x01 national (admin-0) capital
                             bit1 0x02 admin-1 (state/province/region) capital
                             bit2 0x04 megacity        (NE MEGACITY == 1)
                             bit3 0x08 world city      (NE WORLDCITY == 1)
                             bit4 0x10 country label anchor, not a settlement
                             bits 5..7 reserved, currently 0
   22  u8    labelRank       NE LABELRANK, a secondary label-priority hint
                             (lower = label sooner; 255 = unknown)
   23  u8    reserved        = 0
  Byte address of place p = placeTableOffset + p * placeRecBytes.
  Records are SORTED WITHIN EACH LAYER by (rank asc, popMax desc, name asc), so
  "the N most important places of a layer" is simply the first N records of it.

RANK INDEX -- pointLayerCount blocks of rankIndexEntries(=16) u32 at rankIndexOffset.
  Blocks appear in the same order as the point layers appear in the layer table
  (i.e. the k-th layer with geomType==2 uses block k). Block address =
  rankIndexOffset + k * rankIndexEntries * 4.
    rankStart[r] = number of records IN THAT LAYER whose rank is strictly < r.
  So, relative to the layer's firstRecord:
    all places with rank <= R   ->  records [0, rankStart[R+1])          (R <= 14)
    exactly rank R              ->  records [rankStart[R], rankStart[R+1])
    rankStart[0] is always 0; rank 15 is reserved/unused, so rankStart[15] is the
    count of every usable record and equals recordCount in practice.
  This is a convenience only -- it is derivable from the sorted rank bytes.

RANK -> ZOOM.  rank is Natural Earth SCALERANK for places_10m (0..10, 0 = a global
  city) and LABELRANK for country_labels (2..7, 2 = a country you label first).
  Both are "lower = draw earlier". A workable Web-Mercator-tile mapping, using the
  rank index to take a prefix of the layer:
      zoom z (0=whole world) : draw places_10m with rank <= z - 1
      z 0-1  : country_labels rank <= 3      places: none
      z 2    : country_labels rank <= 5      places rank <= 1   (   68)
      z 3    : country_labels all            places rank <= 2   (  186)
      z 4    :                               places rank <= 3   (  522)
      z 5    :                               places rank <= 4   ( 1128)
      z 6    :                               places rank <= 6   ( 2445)
      z 7    :                               places rank <= 7   ( 5527)
      z 8    :                               places rank <= 8   ( 6770)
      z >= 9 :                               all 7342
  (counts in parentheses are this build's rankStart[R+1] for places_10m). Tune the
  thresholds to taste -- nothing in the file depends on them. Within a rank the
  records are already ordered by population, so truncating mid-rank still keeps the
  bigger cities. Capitals (flags bit0) are worth promoting a rank or two.

NAME OFFSET TABLE -- (nameCount + 1) u32 values at nameOffsetTableOffset:
  name i occupies nameBlob bytes [table[i], table[i+1]).  UTF-8, not NUL-terminated.
  String length = table[i+1] - table[i]. Name 0 is the empty string.

NAME BLOB -- nameBlobBytes of concatenated UTF-8 at nameBlobOffset.
  Byte address of name i = nameBlobOffset + table[i].

SECTION ORDER in the file (each starts where the previous ends, all 4-byte aligned):
  header(128) | layer table | feature table | ring table | coord blob |
  place table | rank index | name offset table | name blob
  Do not rely on this order -- always follow the header offsets.

--- Kotlin sketch -------------------------------------------------------------
  val bb = FileChannel.open(p).map(READ_ONLY, 0, size).order(ByteOrder.LITTLE_ENDIAN)
  val featOff = bb.getInt(36); val ringOff = bb.getInt(40); val coordOff = bb.getInt(44)
  val scale = bb.getInt(28).toDouble()
  // feature f:
  val base = featOff + f * 28
  val nameId = bb.getInt(base); val firstRing = bb.getInt(base + 4); val nRings = bb.getInt(base + 8)
  val minLon = bb.getFloat(base + 12) /* ... */
  // ring r of that feature:
  val rb = ringOff + (firstRing + r) * 8
  val firstPt = bb.getInt(rb); val nPts = bb.getInt(rb + 4)
  var q = coordOff + firstPt * 8
  repeat(nPts) { val lon = bb.getInt(q) / scale; val lat = bb.getInt(q + 4) / scale; q += 8 }

  // --- point layers ---
  val placeOff = bb.getInt(72); val placeStride = bb.getInt(76)
  val rankOff  = bb.getInt(80); val rankEntries = bb.getInt(84)
  // walk the layer table to find layer "places_10m"; k = its index among geomType==2 layers
  val firstRec = bb.getInt(layerBase + 4); val nRec = bb.getInt(layerBase + 8)
  fun rankStart(k: Int, r: Int) = bb.getInt(rankOff + (k * rankEntries + r) * 4)
  val visible = if (maxRank >= 14) nRec else rankStart(k, maxRank + 1)   // prefix length
  for (i in 0 until visible) {
      val b = placeOff + (firstRec + i) * placeStride
      val nameId  = bb.getInt(b)
      val lon     = bb.getInt(b + 4) / scale
      val lat     = bb.getInt(b + 8) / scale
      val popMax  = bb.getInt(b + 12).toLong() and 0xFFFFFFFFL
      val ctryId  = bb.getInt(b + 16)
      val rank    = bb.get(b + 20).toInt() and 0xFF
      val flags   = bb.get(b + 21).toInt() and 0xFF
      val isCapital = (flags and 0x01) != 0
      // name: nameBlob[nameTable[nameId] until nameTable[nameId+1]] decoded UTF-8
  }
================================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MAP_DIR = REPO / "mobile" / "assets_build" / "map"
BIN_PATH = MAP_DIR / "world.bin"
PNG_PATH = MAP_DIR / "preview.png"

HEADER_FIELDS = [
    "magic", "version", "layerCount", "featureCount", "ringCount", "pointCount",
    "nameCount", "coordScale", "layerTableOffset", "featureTableOffset",
    "ringTableOffset", "coordOffset", "nameOffsetTableOffset", "nameBlobOffset",
    "nameBlobBytes", "headerBytes",
    # --- v2 only ---
    "pointLayerCount", "placeCount", "placeTableOffset", "placeRecBytes",
    "rankIndexOffset", "rankIndexEntries", "layerRecBytes", "featureRecBytes",
    "ringRecBytes", "coordRecBytes", "fileBytes",
    "reserved0", "reserved1", "reserved2", "reserved3", "reserved4",
]

FEATURE_DTYPE = np.dtype([
    ("name", "<u4"), ("first", "<u4"), ("n", "<u4"),
    ("mnx", "<f4"), ("mny", "<f4"), ("mxx", "<f4"), ("mxy", "<f4"),
])

PLACE_DTYPE = np.dtype([
    ("name", "<u4"), ("lonQ", "<i4"), ("latQ", "<i4"), ("popMax", "<u4"),
    ("country", "<u4"), ("rank", "u1"), ("flags", "u1"), ("labelRank", "u1"),
    ("reserved", "u1"),
])

GEOM_POLYGON, GEOM_LINE, GEOM_POINT = 0, 1, 2
GEOM_NAME = {0: "polygon", 1: "line   ", 2: "point  "}
FLAG_CAPITAL, FLAG_ADM1_CAPITAL = 0x01, 0x02
FLAG_MEGACITY, FLAG_WORLDCITY, FLAG_COUNTRY_LABEL = 0x04, 0x08, 0x10


class WorldMap:
    """Zero-copy reader: everything is a numpy view onto the mmapped file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.buf = np.memmap(path, dtype=np.uint8, mode="r")
        h = np.frombuffer(self.buf[:128].tobytes(), dtype="<u4")
        self.h = dict(zip(HEADER_FIELDS, (int(x) for x in h)))
        if self.h["magic"] != 0x4D57474E:
            raise ValueError(f"bad magic 0x{self.h['magic']:08X}")
        if self.h["version"] not in (1, 2):
            raise ValueError(f"unsupported version {self.h['version']}")
        if self.h["version"] == 1:  # v1 had no point layers and a 64-byte header
            self.h.update({k: 0 for k in HEADER_FIELDS[16:]})
            self.h["headerBytes"] = 64
            self.h["placeRecBytes"], self.h["rankIndexEntries"] = 24, 16
        H = self.h
        self.scale = float(H["coordScale"])

        def view(off, dtype, count):
            return np.frombuffer(self.buf, dtype=dtype, count=count, offset=off)

        self.layers = view(H["layerTableOffset"], "<u4", H["layerCount"] * 4).reshape(-1, 4)
        self.features = view(H["featureTableOffset"], FEATURE_DTYPE, H["featureCount"])
        self.rings = view(H["ringTableOffset"], "<u4", H["ringCount"] * 2).reshape(-1, 2)
        self.coords = view(H["coordOffset"], "<i4", H["pointCount"] * 2).reshape(-1, 2)
        if H["placeCount"]:
            self.places = view(H["placeTableOffset"], PLACE_DTYPE, H["placeCount"])
            self.rank_index = view(H["rankIndexOffset"], "<u4",
                                   H["pointLayerCount"] * H["rankIndexEntries"]
                                   ).reshape(-1, H["rankIndexEntries"])
        else:
            self.places = np.zeros(0, PLACE_DTYPE)
            self.rank_index = np.zeros((0, 16), "<u4")
        self.name_off = view(H["nameOffsetTableOffset"], "<u4", H["nameCount"] + 1)
        blob = self.buf[H["nameBlobOffset"]:H["nameBlobOffset"] + H["nameBlobBytes"]].tobytes()
        self.names = [
            blob[self.name_off[i]:self.name_off[i + 1]].decode("utf-8")
            for i in range(H["nameCount"])
        ]

    # -- accessors ----------------------------------------------------------
    def name(self, nid: int) -> str:
        return self.names[int(nid)]

    def layer(self, layer_name: str):
        for row in self.layers:
            if self.names[int(row[0])] == layer_name:
                return int(row[1]), int(row[2]), int(row[3])
        raise KeyError(layer_name)

    def point_layer_names(self) -> list[str]:
        return [self.names[int(r[0])] for r in self.layers if int(r[3]) == GEOM_POINT]

    def place_layer(self, layer_name: str) -> np.ndarray:
        """The slice of the place table owned by a POINT layer, in rank order."""
        first, count, gt = self.layer(layer_name)
        if gt != GEOM_POINT:
            raise TypeError(f"{layer_name} is not a POINT layer (geomType={gt})")
        return self.places[first:first + count]

    def rank_block(self, layer_name: str) -> np.ndarray:
        """rankStart[] for a POINT layer: count of its records with rank < r."""
        k = self.point_layer_names().index(layer_name)
        return self.rank_index[k]

    def places_up_to_rank(self, layer_name: str, max_rank: int) -> np.ndarray:
        """Exactly what the renderer does: take a prefix of the layer."""
        recs = self.place_layer(layer_name)
        if max_rank >= self.h["rankIndexEntries"] - 1:
            return recs
        return recs[:int(self.rank_block(layer_name)[max_rank + 1])]

    def place_lonlat(self, rec) -> tuple[float, float]:
        return float(rec["lonQ"]) / self.scale, float(rec["latQ"]) / self.scale

    def ring_xy(self, ring_idx: int) -> np.ndarray:
        first, n = self.rings[ring_idx]
        return self.coords[first:first + n].astype(np.float64) / self.scale

    def feature_rings(self, fi: int):
        f = self.features[fi]
        return [self.ring_xy(int(f["first"]) + k) for k in range(int(f["n"]))]


# --------------------------------------------------------------------- geometry
def point_in_ring(lon: float, lat: float, ring: np.ndarray) -> bool:
    """Crossing-number test. Ring is implicitly closed (last -> first)."""
    x, y = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    straddle = (y > lat) != (yj > lat)
    if not straddle.any():
        return False
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x + (lat - y) * (xj - x) / (yj - y)
    return bool(np.count_nonzero(straddle & (lon < xint)) & 1)


def locate(wm: WorldMap, lat: float, lon: float, layer_name: str = "countries_50m"):
    """Strict containment: returns (country|None, featureIndex, nBboxCandidates)."""
    first, count, _ = wm.layer(layer_name)
    f = wm.features[first:first + count]
    cand = np.nonzero(
        (f["mnx"] <= lon) & (lon <= f["mxx"]) & (f["mny"] <= lat) & (lat <= f["mxy"])
    )[0]
    for c in cand:
        fi = first + int(c)
        rings = wm.feature_rings(fi)
        if not point_in_ring(lon, lat, rings[0]):
            continue
        if any(point_in_ring(lon, lat, h) for h in rings[1:]):
            continue  # inside a hole (enclave)
        return wm.name(wm.features[fi]["name"]), fi, len(cand)
    return None, -1, len(cand)


def _seg_dist_km(lat: float, lon: float, ring: np.ndarray) -> float:
    """Min distance point->closed polyline, equirectangular approximation."""
    kx = 111.320 * np.cos(np.radians(lat))
    ky = 110.574
    a = np.column_stack(((ring[:, 0] - lon) * kx, (ring[:, 1] - lat) * ky))
    b = np.roll(a, -1, axis=0)
    d = b - a
    L2 = (d * d).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(-(a * d).sum(1) / np.where(L2 == 0, 1.0, L2), 0.0, 1.0)
    p = a + t[:, None] * d
    return float(np.sqrt((p * p).sum(1)).min())


def locate_snapped(wm: WorldMap, lat: float, lon: float,
                   layer_name: str = "countries_50m", max_km: float = 50.0):
    """Containment, else the nearest country within max_km. Returns (name, km).
    Needed in practice: at 1:50m a harbour-front coordinate (Manhattan, Guanabara
    Bay) sits in the water, so a strict test returns nothing."""
    name, _, _ = locate(wm, lat, lon, layer_name)
    if name is not None:
        return name, 0.0
    first, count, _ = wm.layer(layer_name)
    f = wm.features[first:first + count]
    dlat = max_km / 110.574
    dlon = max_km / max(111.320 * np.cos(np.radians(lat)), 1.0)
    cand = np.nonzero(
        (f["mnx"] - dlon <= lon) & (lon <= f["mxx"] + dlon)
        & (f["mny"] - dlat <= lat) & (lat <= f["mxy"] + dlat)
    )[0]
    best, best_d = None, max_km
    for c in cand:
        fi = first + int(c)
        for r in wm.feature_rings(fi):
            dist = _seg_dist_km(lat, lon, r)
            if dist < best_d:
                best_d, best = dist, wm.name(wm.features[fi]["name"])
    return best, best_d


# ---------------------------------------------------------------------- preview
def render_preview(wm: WorldMap, out: Path, test_points) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.collections import PathCollection, LineCollection

    fig, ax = plt.subplots(figsize=(20, 10), dpi=110)
    ax.set_facecolor("#0d1b2a")

    # filled country polygons (exterior + holes as one compound path per feature)
    paths = []
    first, count, _ = wm.layer("countries_50m")
    for fi in range(first, first + count):
        verts, codes = [], []
        for ring in wm.feature_rings(fi):
            if len(ring) < 3:
                continue
            verts.append(np.vstack([ring, ring[:1]]))
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY])
        if verts:
            paths.append(MplPath(np.concatenate(verts), codes))
    ax.add_collection(PathCollection(paths, facecolors="#3f6d52", edgecolors="#111d16",
                                     linewidths=0.25, zorder=1))

    def lines(layer_name, **kw):
        f0, cn, _ = wm.layer(layer_name)
        segs = []
        for fi in range(f0, f0 + cn):
            for ring in wm.feature_rings(fi):
                if len(ring) >= 2:
                    segs.append(ring)
        ax.add_collection(LineCollection(segs, **kw))
        return len(segs)

    lines("coastline_50m", colors="#9dd6ff", linewidths=0.35, zorder=4)

    # lakes as filled polygons
    f0, cn, _ = wm.layer("lakes_50m")
    lake_paths = []
    for fi in range(f0, f0 + cn):
        for ring in wm.feature_rings(fi):
            if len(ring) >= 3:
                lake_paths.append(MplPath(np.vstack([ring, ring[:1]]),
                                          [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1)
                                          + [MplPath.CLOSEPOLY]))
    ax.add_collection(PathCollection(lake_paths, facecolors="#1b6ca8", edgecolors="none", zorder=2))

    # ---- country label anchors (rank <= 4) -------------------------------------
    n_ctry = 0
    if "country_labels" in wm.point_layer_names():
        for rec in wm.places_up_to_rank("country_labels", 4):
            lon, lat = wm.place_lonlat(rec)
            ax.annotate(wm.name(rec["name"]), (lon, lat), ha="center", va="center",
                        fontsize=5.5, color="#dfe8ef", alpha=0.75, zorder=5)
            n_ctry += 1

    # ---- populated places: dots for rank <= 3, names for rank <= 1 -------------
    n_dot = n_lbl = 0
    if "places_10m" in wm.point_layer_names():
        dots = wm.places_up_to_rank("places_10m", 3)
        lons = dots["lonQ"].astype(np.float64) / wm.scale
        lats = dots["latQ"].astype(np.float64) / wm.scale
        cap = (dots["flags"] & FLAG_CAPITAL) != 0
        ax.scatter(lons[~cap], lats[~cap], s=1.6, c="#ffffff", alpha=0.55,
                   linewidths=0, zorder=5)
        ax.scatter(lons[cap], lats[cap], s=4.0, marker="s", c="#ff9f1c",
                   linewidths=0, zorder=6)
        n_dot = len(dots)
        for rec in wm.places_up_to_rank("places_10m", 1):
            lon, lat = wm.place_lonlat(rec)
            is_cap = bool(rec["flags"] & FLAG_CAPITAL)
            ax.plot(lon, lat, marker="o", ms=2.2, mfc="#fff3b0", mec="#000000",
                    mew=0.3, zorder=7)
            ax.annotate(wm.name(rec["name"]), (lon, lat), textcoords="offset points",
                        xytext=(3, 2), fontsize=6.5, zorder=7,
                        color="#ffd166" if is_cap else "#ffffff",
                        fontweight="bold" if is_cap else "normal")
            n_lbl += 1

    for lat, lon, expected in test_points:
        ax.plot(lon, lat, marker="o", ms=7, mfc="none", mec="#ff477e", mew=1.2, zorder=8)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color="#ffffff", alpha=0.08, lw=0.5)
    ax.set_title(f"world.bin v{wm.h['version']} preview  -  {wm.h['featureCount']} features / "
                 f"{wm.h['pointCount']} vertices / {wm.h['placeCount']} places / "
                 f"{wm.path.stat().st_size/1e6:.2f} MB     "
                 f"[labels: {n_lbl} places rank<=1, {n_dot} dots rank<=3, "
                 f"{n_ctry} countries rank<=4]",
                 color="#e0e1dd", fontsize=11)
    ax.tick_params(colors="#8d99ae", labelsize=8)
    fig.patch.set_facecolor("#0d1b2a")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------------- main
TEST_POINTS = [
    (50.08, 14.44, "Czechia"),
    (35.68, 139.69, "Japan"),
    (-33.87, 151.21, "Australia"),
    (40.71, -74.01, "United States of America"),
    (-1.29, 36.82, "Kenya"),
]


# ------------------------------------------------------------------ place checks
SAMPLE_PLACES = [
    "Tokyo", "London", "New York", "São Paulo", "Cairo", "Sydney", "Moscow",
    "Prague", "Reykjavík", "Nuuk", "Ushuaia", "Timbuktu", "Kathmandu",
    "Yellowknife", "Alice Springs",
]


def check_places(wm: WorldMap) -> list[str]:
    """Validate every POINT layer; returns a list of error strings."""
    errs = []
    total = 0
    for lname in wm.point_layer_names():
        recs = wm.place_layer(lname)
        blk = wm.rank_block(lname)
        total += len(recs)
        lon = recs["lonQ"].astype(np.float64) / wm.scale
        lat = recs["latQ"].astype(np.float64) / wm.scale
        if len(recs) and (lon.min() < -180.001 or lon.max() > 180.001):
            errs.append(f"{lname}: lon out of range [{lon.min()}, {lon.max()}]")
        if len(recs) and (lat.min() < -90.001 or lat.max() > 90.001):
            errs.append(f"{lname}: lat out of range [{lat.min()}, {lat.max()}]")
        if np.any(np.diff(recs["rank"].astype(np.int32)) < 0):
            errs.append(f"{lname}: records are not sorted by rank")
        if np.any(recs["rank"] >= wm.h["rankIndexEntries"] - 1):
            errs.append(f"{lname}: rank >= {wm.h['rankIndexEntries'] - 1} (breaks the rank index)")
        if np.any(recs["reserved"] != 0):
            errs.append(f"{lname}: reserved byte is not 0")
        bad_name = int(np.count_nonzero(
            (recs["name"] == 0) | (recs["name"] >= wm.h["nameCount"])))
        if bad_name:
            errs.append(f"{lname}: {bad_name} records with an empty/OOB nameId")
        if int(np.count_nonzero(recs["country"] >= wm.h["nameCount"])):
            errs.append(f"{lname}: country nameId out of range")
        # rank index must agree with the actual rank bytes
        if int(blk[0]) != 0:
            errs.append(f"{lname}: rankStart[0] != 0")
        for r in range(wm.h["rankIndexEntries"]):
            want = int(np.count_nonzero(recs["rank"] < r))
            if int(blk[r]) != want:
                errs.append(f"{lname}: rankStart[{r}]={int(blk[r])} but {want} records have rank<{r}")
                break
        # the prefix helper must return exactly the records of that rank or lower
        for r in (0, 1, 3, 6, 10):
            pref = wm.places_up_to_rank(lname, r)
            if len(pref) and int(pref["rank"].max()) > r:
                errs.append(f"{lname}: places_up_to_rank({r}) leaks rank {int(pref['rank'].max())}")
    if total != wm.h["placeCount"]:
        errs.append(f"point layers cover {total} records but placeCount={wm.h['placeCount']}")
    return errs


def print_place_samples(wm: WorldMap) -> None:
    recs = wm.place_layer("places_10m")
    names = [wm.name(r["name"]) for r in recs]
    # records are rank-sorted, so the FIRST match is the most important homonym
    # (there are 3 Londons and 2 Sydneys in the 10m set)
    idx: dict[str, int] = {}
    for i, n in enumerate(names):
        idx.setdefault(n, i)
    print("sample places (rank 0 = most important; sorted by rank, then population):")
    print(f"  {'rank':>4} {'flags':>6} {'name':<22} {'country':<26} "
          f"{'lat':>8} {'lon':>9} {'popMax':>10}")
    for want in SAMPLE_PLACES:
        i = idx.get(want)
        if i is None:  # tolerate NE spelling differences
            i = next((k for k, n in enumerate(names) if n.startswith(want[:5])), None)
        if i is None:
            print(f"  {'?':>4} {'':>6} {want:<22} NOT FOUND")
            continue
        r = recs[i]
        lon, lat = wm.place_lonlat(r)
        f = int(r["flags"])
        tag = ("C" if f & FLAG_CAPITAL else ".") + ("1" if f & FLAG_ADM1_CAPITAL else ".") \
            + ("M" if f & FLAG_MEGACITY else ".") + ("W" if f & FLAG_WORLDCITY else ".")
        print(f"  {int(r['rank']):>4} {tag:>6} {wm.name(r['name']):<22} "
              f"{wm.name(r['country']):<26} {lat:>8.3f} {lon:>9.3f} {int(r['popMax']):>10,}")
    print("  flags: C=national capital  1=admin-1 capital  M=megacity  W=world city")


def compare_geometry(new: WorldMap, old: WorldMap) -> list[str]:
    """Prove the polygon/line layers decode identically between two files."""
    errs = []
    old_layers = {old.names[int(r[0])]: r for r in old.layers if int(r[3]) in (0, 1)}
    new_layers = {new.names[int(r[0])]: r for r in new.layers if int(r[3]) in (0, 1)}
    print(f"comparing polygon/line layers against {old.path.name} (v{old.h['version']}):")
    for lname in new_layers:
        if lname not in old_layers:
            errs.append(f"{lname}: missing from the old file")
            continue
        nf0, nfn, ngt = (int(v) for v in new_layers[lname][1:4])
        of0, ofn, ogt = (int(v) for v in old_layers[lname][1:4])
        status = "OK"
        if (nfn, ngt) != (ofn, ogt):
            errs.append(f"{lname}: {ofn} features/gt{ogt} -> {nfn} features/gt{ngt}")
            status = "DIFFERENT"
        else:
            nv = ov = 0
            for k in range(nfn):
                fn, fo = new.features[nf0 + k], old.features[of0 + k]
                if new.name(fn["name"]) != old.name(fo["name"]) or int(fn["n"]) != int(fo["n"]):
                    errs.append(f"{lname}: feature {k} name/ringCount differs")
                    status = "DIFFERENT"
                    break
                if (float(fn["mnx"]), float(fn["mny"]), float(fn["mxx"]), float(fn["mxy"])) != \
                   (float(fo["mnx"]), float(fo["mny"]), float(fo["mxx"]), float(fo["mxy"])):
                    errs.append(f"{lname}: feature {k} bbox differs")
                    status = "DIFFERENT"
                    break
                for r in range(int(fn["n"])):
                    a = new.coords[new.rings[int(fn["first"]) + r][0]:][:new.rings[int(fn["first"]) + r][1]]
                    b = old.coords[old.rings[int(fo["first"]) + r][0]:][:old.rings[int(fo["first"]) + r][1]]
                    if a.shape != b.shape or not np.array_equal(a, b):
                        errs.append(f"{lname}: feature {k} ring {r} coordinates differ")
                        status = "DIFFERENT"
                        break
                    nv += len(a)
                    ov += len(b)
                if status != "OK":
                    break
            print(f"  {lname:<16} {nfn:>5} features, {nv:>7} vertices  bit-identical: {status}")
            continue
        print(f"  {lname:<16} {status}")
    gone = sorted(set(old_layers) - set(new_layers))
    if gone:
        print(f"  layers removed on purpose: {', '.join(gone)}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", metavar="OLD.bin", default=None,
                    help="also verify the polygon/line layers decode identically "
                         "against an older world.bin")
    args = ap.parse_args()

    if not BIN_PATH.exists():
        print(f"missing {BIN_PATH}; run build_map_assets.py first", file=sys.stderr)
        return 2
    wm = WorldMap(BIN_PATH)
    size = BIN_PATH.stat().st_size

    print("=" * 78)
    print(f"file        : {BIN_PATH}")
    print(f"size        : {size:,} bytes ({size/1e6:.2f} MB)")
    print(f"magic/ver   : NGWM / v{wm.h['version']}   coordScale={wm.h['coordScale']} "
          f"(~{111320/wm.h['coordScale']*100:.2f} cm per unit at the equator)")
    print(f"features    : {wm.h['featureCount']:,}")
    print(f"rings       : {wm.h['ringCount']:,}")
    print(f"vertices    : {wm.h['pointCount']:,}   ({wm.h['pointCount']*8/1e6:.2f} MB of coords)")
    print(f"places      : {wm.h['placeCount']:,} in {wm.h['pointLayerCount']} point layer(s)"
          f"   ({wm.h['placeCount']*wm.h['placeRecBytes']/1e6:.2f} MB of records)")
    print(f"names       : {wm.h['nameCount']:,}  ({wm.h['nameBlobBytes']:,} bytes of UTF-8)")
    print("-" * 78)
    for row in wm.layers:
        nid, f0, cn, gt = (int(v) for v in row)
        if gt == GEOM_POINT:
            sub = wm.places[f0:f0 + cn]
            caps = int(np.count_nonzero(sub["flags"] & FLAG_CAPITAL))
            rk = sub["rank"]
            print(f"  {wm.names[nid]:<16} {GEOM_NAME[gt]} places  ={cn:>5} "
                  f"rank={int(rk.min())}..{int(rk.max())} capitals={caps:>4} "
                  f"names={len(set(int(x) for x in sub['name'])):>5}")
        else:
            sub = wm.features[f0:f0 + cn]
            pts = sum(int(wm.rings[int(f["first"]) + k][1])
                      for f in sub for k in range(int(f["n"])))
            rngs = int(sub["n"].sum())
            print(f"  {wm.names[nid]:<16} {GEOM_NAME[gt]} features={cn:>5} rings={rngs:>5} "
                  f"points={pts:>7} names={len(set(int(x) for x in sub['name'])):>5}")
    print("-" * 78)

    # ---- structural checks
    errs = []
    exp_end = wm.h["nameBlobOffset"] + wm.h["nameBlobBytes"]
    if exp_end != size:
        errs.append(f"trailing bytes: header says {exp_end}, file is {size}")
    if wm.h["fileBytes"] != size:
        errs.append(f"header fileBytes={wm.h['fileBytes']} but file is {size}")
    for f, want in [("headerBytes", 128), ("placeRecBytes", 24), ("layerRecBytes", 16),
                    ("featureRecBytes", 28), ("ringRecBytes", 8), ("coordRecBytes", 8),
                    ("rankIndexEntries", 16)]:
        if wm.h[f] != want:
            errs.append(f"header {f}={wm.h[f]}, expected {want}")
    if any(wm.h[f"reserved{i}"] for i in range(5)):
        errs.append("reserved header words are not zero")
    if wm.h["pointLayerCount"] != sum(1 for r in wm.layers if int(r[3]) == GEOM_POINT):
        errs.append("pointLayerCount disagrees with the layer table")
    if any(int(r[3]) > GEOM_POINT for r in wm.layers):
        errs.append("unknown geomType in the layer table")
    if "rivers_50m" in [wm.names[int(r[0])] for r in wm.layers]:
        errs.append("rivers_50m is still present")
    errs += check_places(wm)
    if int(wm.rings[:, 1].sum()) != wm.h["pointCount"]:
        errs.append("ring pointCounts do not sum to header pointCount")
    if int(wm.features["n"].sum()) != wm.h["ringCount"]:
        errs.append("feature ringCounts do not sum to header ringCount")
    lon = wm.coords[:, 0] / wm.scale
    lat = wm.coords[:, 1] / wm.scale
    if lon.min() < -180.001 or lon.max() > 180.001:
        errs.append(f"lon out of range [{lon.min()}, {lon.max()}]")
    if lat.min() < -90.001 or lat.max() > 90.001:
        errs.append(f"lat out of range [{lat.min()}, {lat.max()}]")
    # every feature's bbox must contain all its points
    bad_bbox = 0
    for fi in range(wm.h["featureCount"]):
        f = wm.features[fi]
        for r in wm.feature_rings(fi):
            if (r[:, 0].min() < f["mnx"] or r[:, 0].max() > f["mxx"]
                    or r[:, 1].min() < f["mny"] or r[:, 1].max() > f["mxy"]):
                bad_bbox += 1
                break
    if bad_bbox:
        errs.append(f"{bad_bbox} features have a bbox that clips their own geometry")
    print(f"structural checks : {'OK' if not errs else 'FAILED'}")
    for e in errs:
        print(f"   ! {e}")
    print(f"lon range   : [{lon.min():.4f}, {lon.max():.4f}]   "
          f"lat range: [{lat.min():.4f}, {lat.max():.4f}]")
    print("-" * 78)

    # ---- point-in-polygon
    print("point-in-polygon tests (layer countries_50m):")
    print("  strict = ring containment only;  snapped = nearest country within 50 km")
    ok = ok_snap = 0
    for lat_, lon_, expected in TEST_POINTS:
        got, fi, ncand = locate(wm, lat_, lon_)
        snap, km = locate_snapped(wm, lat_, lon_)
        ok += got == expected
        ok_snap += snap == expected
        note = "OK" if got == expected else (
            f"water at 1:50m -> snapped to {snap} ({km:.1f} km)" if snap == expected
            else f"WRONG (snapped: {snap})")
        print(f"  ({lat_:>7.2f},{lon_:>8.2f})  strict={str(got):<26} "
              f"expected={expected:<26} {note}   [{ncand} bbox cand]")
    print("extra sanity points (strict / snapped):")
    for lat_, lon_, label in [(0.0, -30.0, "mid-Atlantic, 1600 km offshore"),
                              (48.86, 2.35, "Paris"),
                              (-22.91, -43.17, "Rio de Janeiro"),
                              (55.75, 37.62, "Moscow"),
                              (28.61, 77.21, "New Delhi"),
                              (1.35, 103.82, "Singapore"),
                              (64.15, -21.94, "Reykjavik")]:
        got, _, _ = locate(wm, lat_, lon_)
        snap, km = locate_snapped(wm, lat_, lon_)
        print(f"  ({lat_:>7.2f},{lon_:>8.2f})  -> {str(got):<26} / "
              f"{str(snap):<26} {km:5.1f} km   ({label})")
    print("-" * 78)

    # ---- place layers
    print_place_samples(wm)
    print("-" * 78)
    blk = wm.rank_block("places_10m")
    n_places = len(wm.place_layer("places_10m"))
    print("rank -> prefix length (places_10m), i.e. how many records to draw:")
    print("  " + "  ".join(
        f"<={r}:{(int(blk[r+1]) if r + 1 < len(blk) else n_places):>5}" for r in range(0, 11)))
    cblk = wm.rank_block("country_labels")
    n_ctry = len(wm.place_layer("country_labels"))
    print("rank -> prefix length (country_labels):")
    print("  " + "  ".join(
        f"<={r}:{(int(cblk[r+1]) if r + 1 < len(cblk) else n_ctry):>5}" for r in range(0, 8)))
    caps = wm.place_layer("places_10m")
    caps = caps[(caps["flags"] & FLAG_CAPITAL) != 0]
    print(f"national capitals : {len(caps)}   megacities: "
          f"{int(np.count_nonzero(wm.place_layer('places_10m')['flags'] & FLAG_MEGACITY))}")
    # country label anchors must land inside their own country
    outside = []
    for rec in wm.place_layer("country_labels"):
        lon, lat = wm.place_lonlat(rec)
        got, _, _ = locate(wm, lat, lon, "countries_50m")
        if got != wm.name(rec["name"]):
            outside.append((wm.name(rec["name"]), got))
    print(f"country label anchors inside their own polygon: "
          f"{n_ctry - len(outside)}/{n_ctry}")
    if outside:
        print("   ! " + ", ".join(f"{a} -> {b}" for a, b in outside[:8]))
        errs.append(f"{len(outside)} country label anchors are not inside their country")
    print("-" * 78)

    cmp_errs = []
    if args.compare:
        cmp_errs = compare_geometry(wm, WorldMap(Path(args.compare)))
        print(f"geometry comparison : {'IDENTICAL' if not cmp_errs else 'FAILED'}")
        for e in cmp_errs:
            print(f"   ! {e}")
        print("-" * 78)

    render_preview(wm, PNG_PATH, TEST_POINTS)
    print(f"preview     : {PNG_PATH}  ({PNG_PATH.stat().st_size/1e6:.2f} MB)")
    print(f"PIP result  : strict {ok}/{len(TEST_POINTS)} correct, "
          f"with 50 km coastal snap {ok_snap}/{len(TEST_POINTS)} correct")
    print(f"file size   : {size:,} bytes ({size/1e6:.2f} MB)  budget 4.00 MB -> "
          f"{'OK' if size < 4e6 else 'OVER'}")
    print("=" * 78)
    return 0 if (ok >= 4 and ok_snap == len(TEST_POINTS) and not errs
                 and not cmp_errs and size < 4e6) else 1


if __name__ == "__main__":
    sys.exit(main())

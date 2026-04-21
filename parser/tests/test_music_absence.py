"""Regression tests for the Music chunks (probe #11) — absence antibody.

FNAF 1 ships without any Music bank. The Clickteam Fusion 2.5 runtime
allocates two chunk ids for the Music genre:

- 0x5558 MusicOffsets  (sibling of 0x5555 Images, 0x5556 Fonts, 0x5557
  Sounds — a top-level handle → byte-offset table).
- 0x6669 Music         (sibling of 0x6666 Images, 0x6667 Fonts, 0x6668
  Sounds — the actual music-bank body keyed by those offsets).

Scott Cawthon's MMF2 project for FNAF 1 has no music tracks (the in-game
ambience is implemented via sound-effect loops inside 0x6668), so neither
chunk exists in the pack. Probe #11 pins that absence instead of writing
a full decoder pair: shipping `decode_music_offsets` + `decode_music`
decoders that would never be exercised against the only input file we
have is worse than documenting the empirical fact loudly.

This isn't a tautological test. The chunk ids are *known* to the walker
— see ``fnaf_parser.chunk_ids.CHUNK_NAMES`` entries for 0x5558 and
0x6669 — so a future re-pack of FNAF 1 that introduced music tracks
would pass the walker's strict-unknown gate cleanly and these tests
would fire. The walker's failure mode for an unknown id is a
`ChunkWalkError`; the failure mode for an *unexpectedly present* music
chunk is this antibody. Two different gates, two different signals.

Antibody coverage:

- Absence #1        : 0x5558 MusicOffsets is not emitted by the walker.
- Absence #2        : 0x6669 Music is not emitted by the walker.
- Inventory pin     : the asset-bank chunk id set is exactly
                      {0x5555, 0x5556, 0x5557, 0x6666, 0x6667, 0x6668}.
                      Any drift — a missing bank, an added bank, a
                      renamed bank — changes the set and fires loudly.

This also closes out the envelope phase of the Asset Extraction arc:
every asset-bank chunk FNAF 1 actually contains now has a decoder and a
snapshot. The next probe (#7.1) pivots from envelopes to payloads —
decoding per-image pixel bytes inside the 0x6666 Images bank.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fnaf_parser.chunk_walker import walk_chunks
from fnaf_parser.pe_walker import FNAF1_DATA_PACK_START

FNAF_EXE = Path(__file__).resolve().parent.parent.parent / "FiveNightsatFreddys.exe"

# Music-genre chunk ids known to the Clickteam Fusion 2.5 runtime. Both
# are present in `chunk_ids.CHUNK_NAMES` — the walker recognises them
# structurally, so their absence here is a real empirical claim, not a
# side-effect of strict-unknown rejection.
MUSIC_OFFSETS_CHUNK_ID = 0x5558
MUSIC_CHUNK_ID = 0x6669

# The exact asset-bank chunk id set FNAF 1's pack carries. Pinned as a
# set (not a list) because chunk ordering inside the pack is not part of
# the invariant — only membership is.
#
# - 0x5555 ImageOffsets  / 0x6666 Images
# - 0x5556 FontOffsets   / 0x6667 Fonts
# - 0x5557 SoundOffsets  / 0x6668 Sounds
FNAF1_ASSET_BANK_CHUNK_IDS = frozenset(
    {0x5555, 0x5556, 0x5557, 0x6666, 0x6667, 0x6668}
)


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_has_no_music_offsets_chunk():
    """Absence #1: the walker must not emit a 0x5558 MusicOffsets record.

    A positive hit here means either (a) the pack has grown a music
    bank — in which case we need a `decode_music_offsets` decoder and
    the three-constant `FNAF1_BANK_OFFSET_DELTA` handshake re-verified
    against it — or (b) chunk-id drift is silently reclassifying some
    other chunk as MusicOffsets. Either way, stop and investigate.
    """
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    seen_ids = {r.id for r in result.records}
    assert MUSIC_OFFSETS_CHUNK_ID not in seen_ids, (
        f"FNAF 1 unexpectedly carries a 0x{MUSIC_OFFSETS_CHUNK_ID:04X} "
        f"MusicOffsets chunk — decoder work needed before this test is "
        f"allowed to pass."
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_has_no_music_chunk():
    """Absence #2: the walker must not emit a 0x6669 Music record.

    Symmetric to the MusicOffsets absence above. A positive hit means a
    music bank has appeared and needs its own decoder + snapshot before
    the asset-extraction pipeline is complete.
    """
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    seen_ids = {r.id for r in result.records}
    assert MUSIC_CHUNK_ID not in seen_ids, (
        f"FNAF 1 unexpectedly carries a 0x{MUSIC_CHUNK_ID:04X} Music "
        f"chunk — decoder work needed before this test is allowed to "
        f"pass."
    )


@pytest.mark.skipif(not FNAF_EXE.exists(), reason="FNAF 1 binary not on disk")
def test_fnaf1_asset_bank_chunk_inventory():
    """Inventory pin: the set of asset-bank ids is exactly the six we
    expect. Not a subset check, not a superset check — equality.

    Failure modes this catches:

    - A future pack loses one of the six (e.g. no fonts) → missing id.
    - A future pack gains a seventh (e.g. 0x6669 Music appears) →
      extra id. The two absence tests above fire on this case too, but
      this inventory assertion is the one-stop shop: any drift in the
      asset-bank topology prints the exact offending set diff.
    - A future walker regression duplicates a chunk into the wrong id
      slot → extra id of some unexpected value.

    The comparison is done on the intersection of seen_ids with the
    full asset-bank id space (0x5555-0x5558 and 0x6666-0x6669), so
    non-asset chunks (headers, strings, frames, etc.) don't drown the
    diff.
    """
    result = walk_chunks(FNAF_EXE, pack_start=FNAF1_DATA_PACK_START)
    seen_ids = {r.id for r in result.records}
    asset_bank_id_space = frozenset(
        list(range(0x5555, 0x5559)) + list(range(0x6666, 0x666A))
    )
    asset_bank_ids_seen = seen_ids & asset_bank_id_space
    assert asset_bank_ids_seen == FNAF1_ASSET_BANK_CHUNK_IDS, (
        f"FNAF 1 asset-bank inventory drift: "
        f"expected {sorted(hex(i) for i in FNAF1_ASSET_BANK_CHUNK_IDS)}, "
        f"saw {sorted(hex(i) for i in asset_bank_ids_seen)}"
    )

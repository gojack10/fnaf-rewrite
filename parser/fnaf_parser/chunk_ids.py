"""Known Clickteam Fusion 2.5 chunk IDs, cross-checked against two references.

Sources:
- CTFAK2.0  C#  reference/CTFAK2.0/Core/CTFAK.Core/CCN/Chunks/ChunkList.cs
              (newer, proper superset, 82 entries)
- Anaconda  Py  reference/Anaconda/mmfparser/data/chunk.pyx
              (older, 69 entries, subset of CTFAK)

When CTFAK and Anaconda disagree on a name, the CTFAK human-friendly name wins
(it's the authoritative spec per the Asset Extraction plan) but we keep the
Anaconda symbolic name in `ANACONDA_NAMES` for cross-checking. IDs that only
appear in CTFAK are flagged via `DUAL_CONFIRMED` (False for CTFAK-only, True
for both).

This module is the "strict mode" gate for Parser Antibody #1: a chunk_id not
present in `CHUNK_NAMES` is an error, not a silent skip.
"""

from __future__ import annotations

# Human-readable names (from CTFAK2.0 ChunkList.cs). Superset of Anaconda.
# id -> human name
CHUNK_NAMES: dict[int, str] = {
    0x1122: "Preview",
    0x2222: "Mini-Header",
    0x2223: "Header",
    0x2224: "Name",
    0x2225: "Author",
    0x2226: "Menu",
    0x2227: "Extension Path",
    0x2228: "Extensions",
    0x2229: "Frame Items",
    0x222A: "Global Events",
    0x222B: "Frame Handles",
    0x222C: "Extension Data",
    0x222D: "Additional Extension",
    0x222E: "App Editor-Filename",
    0x222F: "App Target-Filename",
    0x2230: "App Docs",
    0x2231: "Other Extensions",
    0x2232: "Global Values",
    0x2233: "Global Strings",
    0x2234: "Extension List",
    0x2235: "Icon 16x16",
    0x2236: "Is Demo",
    0x2237: "Serial Number",
    0x2238: "Binary Files",
    0x2239: "Menu Images",
    0x223A: "About Text",
    0x223B: "Copyright",
    0x223C: "Global Value Names",
    0x223D: "Global String Names",
    0x223E: "Movement Extensions",
    0x223F: "Frame Items 2",
    0x2240: "Exe Only",
    0x2242: "Protection",
    0x2243: "Shaders",
    0x2245: "Extended Header",
    0x2246: "Spacer",
    0x2247: "Frame Offset",
    0x2248: "Ad Mob ID",
    0x224B: "Android Menu",
    0x2253: "2.5+ Object Headers",
    0x2254: "2.5+ Object Names",
    0x2255: "2.5+ Object Shaders",
    0x2256: "2.5+ Object Properties",
    0x2258: "Font Info",
    0x2259: "Fonts",
    0x225A: "Shaders (2.5+)",  # CTFAK labels this "Shaders" again; disambiguated here.
    0x3333: "Frame",
    0x3334: "Frame Header",
    0x3335: "Frame Name",
    0x3336: "Frame Password",
    0x3337: "Frame Palette",
    0x3338: "Frame Item Instances",
    0x3339: "Frame Fade In Frame",
    0x333A: "Frame Fade Out Frame",
    0x333B: "Frame Fade In",
    0x333C: "Frame Fade Out",
    0x333D: "Frame Events",
    0x333E: "Frame Play Header",
    0x333F: "Additional Frame Item",
    0x3340: "Additional Frame Item Instance",
    0x3341: "Frame Layers",
    0x3342: "Frame Virtual Rect",
    0x3343: "Demo File Path",
    0x3344: "Random Seed",
    0x3345: "Frame Layer Effects",
    0x3346: "Blu-Ray Frame Options",
    0x3347: "Mvt Timer Base",
    0x3348: "Mosaic Image Table",
    0x3349: "Frame Effects",
    0x334A: "Frame iPhone Options",
    0x4444: "Object Info Header",
    0x4445: "Object Info Name",
    0x4446: "Object Common",
    0x4447: "Unknown Object Chunk",
    0x4448: "Object Effects",
    0x4500: "Object Shapes",
    0x5555: "Image Handles",
    0x5556: "Font Handles",
    0x5557: "Sound Handles",
    0x5558: "Music Handles",
    0x6665: "Bank Offsets",
    0x6666: "Images",
    0x6667: "Fonts",
    0x6668: "Sounds",
    0x6669: "Music",
    0x7EEE: "Fusion 3 Seed",
    0x7F7F: "Last Chunk",
}

# Anaconda's symbolic constant names (uppercase, underscored) — kept for
# cross-referencing and to make "where does this chunk's decoder live in the
# Anaconda source tree" answerable at a glance.
ANACONDA_NAMES: dict[int, str] = {
    0x1122: "PREVIEW",
    0x2222: "APPMINIHEADER",
    0x2223: "APPHEADER",
    0x2224: "APPNAME",
    0x2225: "APPAUTHOR",
    0x2226: "APPMENU",
    0x2227: "EXTPATH",
    0x2228: "EXTENSIONS",
    0x2229: "FRAMEITEMS",
    0x222A: "GLOBALEVENTS",
    0x222B: "FRAMEHANDLES",
    0x222C: "EXTDATA",
    0x222D: "ADDITIONAL_EXTENSION",
    0x222E: "APPEDITORFILENAME",
    0x222F: "APPTARGETFILENAME",
    0x2230: "APPDOC",
    0x2231: "OTHEREXTS",
    0x2232: "GLOBALVALUES",
    0x2233: "GLOBALSTRINGS",
    0x2234: "EXTENSIONS2",
    0x2235: "APPICON_16x16x8",
    0x2236: "DEMOVERSION",
    0x2237: "SECNUM",
    0x2238: "BINARYFILES",
    0x2239: "APPMENUIMAGES",
    0x223A: "ABOUTTEXT",
    0x223B: "COPYRIGHT",
    0x223C: "GLOBALVALUENAMES",
    0x223D: "GLOBALSTRINGNAMES",
    0x223E: "MVTEXTS",
    0x223F: "FRAMEITEMS_2",
    0x2240: "EXEONLY",
    0x2242: "PROTECTION",
    0x2243: "SHADERS",
    0x2245: "APPHEADER2",
    0x3333: "FRAME",
    0x3334: "FRAMEHEADER",
    0x3335: "FRAMENAME",
    0x3336: "FRAMEPASSWORD",
    0x3337: "FRAMEPALETTE",
    0x3338: "FRAMEITEMINSTANCES",
    0x3339: "FRAMEFADEINFRAME",
    0x333A: "FRAMEFADEOUTFRAME",
    0x333B: "FRAMEFADEIN",
    0x333C: "FRAMEFADEOUT",
    0x333D: "FRAMEEVENTS",
    0x333E: "FRAMEPLAYHEADER",
    0x333F: "ADDITIONAL_FRAMEITEM",
    0x3340: "ADDITIONAL_FRAMEITEMINSTANCE",
    0x3341: "FRAMELAYERS",
    0x3342: "FRAMEVIRTUALRECT",
    0x3343: "DEMOFILEPATH",
    0x3344: "RANDOMSEED",
    0x3345: "FRAMELAYEREFFECTS",
    0x3346: "BLURAYFRAMEOPTIONS",
    0x3347: "MVTTIMERBASE",
    0x3348: "MOSAICIMAGETABLE",
    0x3349: "FRAMEEFFECTS",
    0x334A: "FRAME_IPHONE_OPTIONS",
    0x4444: "OBJINFOHEADER",
    0x4445: "OBJINFONAME",
    0x4446: "OBJECTSCOMMON",
    0x4447: "OBJECTUNKNOWN",
    0x4448: "OBJECTEFFECTS",
    0x5555: "IMAGESOFFSETS",
    0x5556: "FONTSOFFSETS",
    0x5557: "SOUNDSOFFSETS",
    0x5558: "MUSICSOFFSETS",
    0x6666: "IMAGES",
    0x6667: "FONTS",
    0x6668: "SOUNDS",
    0x6669: "MUSICS",
    0x7F7F: "LAST",
}

# Chunks seen in FNAF 1 binaries but absent from both references. CTFAK's
# GUI source comments a bare "//224F" between Spacer (0x2246) and Frame
# Handles (0x222B) in its app-metadata tree without naming or decoding it;
# Antibody #1 surfaced it the first time we walked FNAF 1. Anything added
# here is a probe #4 priority-zero decode target.
EMPIRICAL_NAMES: dict[int, str] = {
    # Seen in FNAF 1 between "Extended Header" (0x2245) and "Menu" (0x2226)
    # in the app-metadata region. CTFAK GUI source comments a bare "//224F"
    # in the same tree position without naming it.
    0x224F: "Unknown App Chunk (0x224F)",
    # Seen once in FNAF 1 between "Exe Only" (0x2240) and the first Frame
    # (0x3333) — size 8, flags 0x0000 (uncompressed). Zero references
    # anywhere; pure empirical discovery. Probe #4 decode priority.
    0x224D: "Unknown App Chunk (0x224D)",
}

# Merge empirical IDs into the master known set. We keep CHUNK_NAMES as the
# single strict-mode gate so is_known() stays O(1) and simple.
CHUNK_NAMES.update(EMPIRICAL_NAMES)

# Chunks found in BOTH references — highest-confidence naming.
DUAL_CONFIRMED: frozenset[int] = frozenset(ANACONDA_NAMES.keys() & CHUNK_NAMES.keys())

# Chunks present only in CTFAK (newer additions). Flag loudly if encountered
# without human cross-check later.
CTFAK_ONLY: frozenset[int] = frozenset(
    CHUNK_NAMES.keys() - ANACONDA_NAMES.keys() - EMPIRICAL_NAMES.keys()
)

# Chunks seen empirically but not in any reference — decode-investigate-first.
EMPIRICAL: frozenset[int] = frozenset(EMPIRICAL_NAMES.keys())

# Sentinel IDs with semantic meaning.
LAST_CHUNK_ID = 0x7F7F


def is_known(chunk_id: int) -> bool:
    """Parser Antibody #1 gate: strict-mode check for chunk IDs."""
    return chunk_id in CHUNK_NAMES


def chunk_label(chunk_id: int) -> str:
    """Human-friendly label used in logs and JSON output.

    Includes hex id + name + confidence marker:
    - `dual` when both references agree
    - `ctfak-only` when only CTFAK defines it (newer 2.5+ chunks)
    - `empirical` when neither reference names it but we saw it in a binary
    - `unknown` when it's not even in the empirical set (Antibody #1 fires)
    """
    name = CHUNK_NAMES.get(chunk_id)
    if name is None:
        return f"0x{chunk_id:04X} (unknown)"
    if chunk_id in EMPIRICAL:
        marker = "empirical"
    elif chunk_id in DUAL_CONFIRMED:
        marker = "dual"
    else:
        marker = "ctfak-only"
    return f"0x{chunk_id:04X} {name} [{marker}]"

# Ported verbatim from reference/Anaconda/mmfparser/data/chunkloaders/parameters/names.py
# Anaconda is (c) Mathias Kaerlev 2012, GPLv3.
"""EventParameter type code → human-name lookup table.

Source: `mmfparser/data/chunkloaders/parameters/names.py`.

Flat map of parameter-type IDs to Clickteam's internal name strings.
These are the `parameter_type` values that appear in every
EventParameter's header (see probe #4.13 EventParameter payload decode);
the payload layout is determined by this ID.

The 15 codes that actually appear in FNAF 1 are a strict subset:
`{1, 2, 6, 9, 10, 14, 16, 22, 23, 25, 26, 27, 32, 45, 50}`. The rest are
kept for completeness so name-resolution of arbitrary packs still works.

Note: Anaconda's upstream file also defines `getName(id)` as a helper.
We don't port the helper — name lookup in this codebase is a single
indexing operation with a loud `KeyError` on unknowns, which is exactly
the "loud failure on unknown IDs" contract the name resolver needs.
"""

from __future__ import annotations

# Flat map: parameter_type_id -> Clickteam internal name string
PARAMETER_NAMES: dict[int, str] = {
    1: "OBJECT",
    2: "TIME",
    3: "SHORT",
    4: "SHORT",
    5: "INT",
    6: "SAMPLE",
    7: "SAMPLE",
    9: "CREATE",
    10: "SHORT",
    11: "SHORT",
    12: "SHORT",
    13: "Every",
    14: "KEY",
    15: "EXPRESSION",
    16: "POSITION",
    17: "JOYDIRECTION",
    18: "SHOOT",
    19: "ZONE",
    21: "SYSCREATE",
    22: "EXPRESSION",
    23: "COMPARISON",
    24: "COLOUR",
    25: "BUFFER4",
    26: "FRAME",
    27: "SAMLOOP",
    28: "MUSLOOP",
    29: "NEWDIRECTION",
    31: "TEXTNUMBER",
    32: "Click",
    33: "PROGRAM",
    34: "OLDPARAM_VARGLO",
    35: "CNDSAMPLE",
    36: "CNDMUSIC",
    37: "REMARK",
    38: "GROUP",
    39: "GROUPOINTER",
    40: "FILENAME",
    41: "STRING",
    42: "CMPTIME",
    43: "PASTE",
    44: "VMKEY",
    45: "EXPSTRING",
    46: "CMPSTRING",
    47: "INKEFFECT",
    48: "MENU",
    49: "GlobalValue",
    50: "AlterableValue",
    51: "FLAG",
    52: "VARGLOBAL_EXP",
    53: "AlterableValueExpression",
    54: "FLAG_EXP",
    55: "EXTENSION",
    56: "8DIRECTIONS",
    57: "MVT",
    58: "GlobalString",
    59: "STRINGGLOBAL_EXP",
    60: "PROGRAM2",
    61: "ALTSTRING",
    62: "ALTSTRING_EXP",
    63: "FILENAME",
    64: "FASTLOOPNAME",
    65: "CHAR_ENCODING_INPUT",
    66: "CHAR_ENCODING_OUTPUT",
}

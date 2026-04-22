# Ported verbatim from reference/Anaconda/mmfparser/data/chunkloaders/objectinfo.pyx
# (lines 25-45). Anaconda is (c) Mathias Kaerlev 2012, GPLv3.
"""Object-type ID → human-name lookup table.

Source: `mmfparser/data/chunkloaders/objectinfo.pyx` lines 25-45.

Flat map of Clickteam's object-type IDs to short human names. The
negative IDs (-7..-1) are built-in event sources (Player, Keyboard,
Create, Timer, Game, Speaker, System). The 0-9 range is the per-type
slot for built-in object classes (QuickBackdrop through SubApplication).

Any object whose type ID is `>= EXTENSION_BASE` (32) is a user
extension — Anaconda's upstream `getObjectType(id)` returns the literal
string "Extension" for those and defers further naming to the
extensionDict layer. We mirror that contract: callers check
`id >= EXTENSION_BASE` first, and only look up in `OBJECT_TYPE_NAMES`
if the ID is below the extension threshold.
"""

from __future__ import annotations

# Any object_type_id >= EXTENSION_BASE is a user extension, not a built-in.
EXTENSION_BASE: int = 32

# Flat map: object_type_id -> Clickteam short name
OBJECT_TYPE_NAMES: dict[int, str] = {
    -7: "Player",
    -6: "Keyboard",
    -5: "Create",
    -4: "Timer",
    -3: "Game",
    -2: "Speaker",
    -1: "System",
    0: "QuickBackdrop",
    1: "Backdrop",
    2: "Active",
    3: "Text",
    4: "Question",
    5: "Score",
    6: "Lives",
    7: "Counter",
    8: "RTF",
    9: "SubApplication",
}

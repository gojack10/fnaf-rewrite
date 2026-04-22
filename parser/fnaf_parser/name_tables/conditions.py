# Ported verbatim from reference/Anaconda/mmfparser/data/chunkloaders/conditions/names.py
# Anaconda is (c) Mathias Kaerlev 2012, GPLv3. Original filenames preserved in
# the module docstring so a reviewer can diff them byte-for-byte against the
# reference tree.
"""Condition code → human-name lookup tables.

Source: `mmfparser/data/chunkloaders/conditions/names.py`.

- `CONDITION_SYSTEM_NAMES` mirrors Anaconda's `systemDict` — nested
  `{object_type_id: {code: name}}`. Built-in object types (negative IDs
  like -1 System, -2 Speaker, ..., -7 Player) and the 0-9 per-type slots
  (Active=2, Question=4, Counter=7, SubApplication=9) each scope their
  own `code → name` table.
- `CONDITION_EXTENSION_NAMES` mirrors Anaconda's flat `extensionDict` —
  the generic object-extension conditions (animation, collision,
  position, alterable values, flags, ...). These apply to any extension
  object (`object_type >= 32`, i.e. above `EXTENSION_BASE`).
"""

from __future__ import annotations

# Nested map: object_type_id -> condition_code -> name
CONDITION_SYSTEM_NAMES: dict[int, dict[int, str]] = {
    2: {
        -81: "ObjectClicked",  # SPRCLICK
    },
    4: {
        -83: "AnswerMatches",
        -82: "AnswerFalse",
        -81: "AnswerTrue",
    },
    7: {
        -81: "CompareCounter",
    },
    9: {
        -84: "SubApplicationPaused",
        -83: "SubApplicationVisible",
        -82: "SubApplicationFinished",
        -81: "SubApplicationFrameChanged",
    },
    -2: {
        -1: "SampleNotPlaying",
        -9: "ChannelPaused",
        -8: "ChannelNotPlaying",
        -7: "MusicPaused",
        -6: "SamplePaused",
        -5: "MusicFinished",
        -4: "NoMusicPlaying",
        -3: "NoSamplesPlaying",
        -2: "SpecificMusicNotPlaying",
    },
    -7: {
        -1: "PLAYERPLAYING",
        -6: "PlayerKeyDown",
        -5: "PlayerDied",
        -4: "PlayerKeyPressed",
        -3: "NumberOfLives",
        -2: "CompareScore",
    },
    -6: {
        -2: "KeyDown",
        -12: "MouseWheelDown",
        -11: "MouseWheelUp",
        -10: "MouseVisible",
        -9: "AnyKeyPressed",
        -8: "WhileMousePressed",
        -7: "ObjectClicked",
        -6: "MouseClickedInZone",
        -5: "MouseClicked",
        -4: "MouseOnObject",
        -3: "MouseInZone",
        -1: "KeyPressed",
    },
    -5: {
        -2: "AllObjectsInZone",  # AllObjectsInZone_Old
        -23: "PickObjectsInLine",
        -22: "PickFlagOff",
        -21: "PickFlagOn",
        -20: "PickAlterableValue",
        -19: "PickFromFixed",
        -18: "PickObjectsInZone",
        -17: "PickRandomObject",
        -16: "PickRandomObjectInZone",
        -15: "CompareObjectCount",
        -14: "AllObjectsInZone",
        -13: "NoAllObjectsInZone",
        -12: "PickFlagOff",  # PickFlagOff_Old
        -11: "PickFlagOn",  # PickFlagOn_Old
        -8: "PickAlterableValue",  # PickAlterableValue_Old
        -7: "PickFromFixed",  # PickFromFixed_Old
        -6: "PickObjectsInZone",  # PickObjectsInZone_Old
        -5: "PickRandomObject",  # PickRandomObject_Old
        -4: "PickRandomObjectInZoneOld",
        -3: "CompareObjectCount",  # CompareObjectCount_Old
        -1: "NoAllObjectsInZone",  # NoAllObjectsInZone_Old
    },
    -4: {
        -8: "Every",
        -7: "TimerEquals",
        -6: "OnTimerEvent",
        -5: "CompareAwayTime",
        -4: "Every",
        -3: "TimerEquals",
        -2: "TimerLess",
        -1: "TimerGreater",
    },
    -3: {
        -1: "StartOfFrame",
        -10: "FrameSaved",
        -9: "FrameLoaded",
        -8: "ApplicationResumed",
        -7: "VsyncEnabled",
        -6: "IsLadder",
        -5: "IsObstacle",
        -4: "EndOfApplication",
        -3: "LEVEL",
        -2: "EndOfFrame",
    },
    -1: {
        -1: "Always",
        -2: "Never",
        -3: "Compare",
        -4: "RestrictFor",
        -5: "Repeat",
        -6: "Once",
        -7: "NotAlways",
        -8: "CompareGlobalValue",
        -9: "Remark",
        -10: "NewGroup",
        -11: "GroupEnd",
        -12: "GroupActivated",
        -13: "RECORDKEY",
        -14: "MenuSelected",
        -15: "FilesDropped",
        -16: "OnLoop",
        -17: "MenuChecked",
        -18: "MenuEnabled",
        -19: "MenuVisible",
        -20: "CompareGlobalString",
        -21: "CloseSelected",
        -22: "ClipboardDataAvailable",
        -23: "OnGroupActivation",
        -24: "OrFiltered",
        -25: "OrLogical",
        -26: "Chance",
        -27: "ElseIf",
        -28: "CompareGlobalValueIntEqual",
        -29: "CompareGlobalValueIntNotEqual",
        -30: "CompareGlobalValueIntLessEqual",
        -31: "CompareGlobalValueIntLess",
        -32: "CompareGlobalValueIntGreaterEqual",
        -33: "CompareGlobalValueIntGreater",
        -34: "CompareGlobalValueDoubleEqual",
        -35: "CompareGlobalValueDoubleNotEqual",
        -36: "CompareGlobalValueDoubleLessEqual",
        -37: "CompareGlobalValueDoubleLess",
        -38: "CompareGlobalValueDoubleGreaterEqual",
        -39: "CompareGlobalValueDoubleGreater",
        -40: "RunningAs",
    },
}


# Flat map: condition_code -> name, for generic object-extension conditions.
CONDITION_EXTENSION_NAMES: dict[int, str] = {
    -1: "AnimationFrame",
    -2: "AnimationFinished",
    -3: "AnimationPlaying",
    -4: "IsOverlapping",
    -5: "Reversed",
    -6: "Bouncing",
    -7: "MovementStopped",
    -8: "FacingInDirection",
    -9: "InsidePlayfield",
    -10: "OutsidePlayfield",
    -11: "EnteringPlayfield",
    -12: "LeavingPlayfield",
    -13: "OnBackgroundCollision",
    -14: "OnCollision",
    -15: "CompareSpeed",
    -16: "CompareY",
    -17: "CompareX",
    -18: "CompareDeceleration",
    -19: "CompareAcceleration",
    -20: "NodeReached",
    -21: "PathFinished",
    -22: "NearWindowBorder",
    -23: "IsOverlappingBackground",
    -24: "FlagOff",
    -25: "FlagOn",
    -26: "CompareFixedValue",
    -27: "CompareAlterableValue",
    -28: "ObjectInvisible",
    -29: "ObjectVisible",
    -30: "ObjectsInZone",
    -31: "NoObjectsInZone",
    -32: "NumberOfObjects",
    -33: "AllDestroyed",
    -34: "PickRandom",
    -35: "NamedNodeReached",
    -36: "CompareAlterableString",
    -37: "IsBold",
    -38: "IsItalic",
    -39: "IsUnderline",
    -40: "IsStrikeOut",
    -41: "OnObjectLoop",
    -42: "CompareAlterableValue",
    -43: "CompareAlterableValue",
}

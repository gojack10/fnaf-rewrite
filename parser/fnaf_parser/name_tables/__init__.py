"""Clickteam ID → human-name lookup tables.

Pure-data port of the 5 name tables from Anaconda's `mmfparser`:

    conditions.CONDITION_SYSTEM_NAMES     # nested {object_type_id: {code: name}}
    conditions.CONDITION_EXTENSION_NAMES  # flat {code: name}
    actions.ACTION_SYSTEM_NAMES           # nested
    actions.ACTION_EXTENSION_NAMES        # flat
    expressions.EXPRESSION_SYSTEM_NAMES   # nested
    expressions.EXPRESSION_EXTENSION_NAMES  # flat
    parameters.PARAMETER_NAMES            # flat {param_type: name}
    object_types.OBJECT_TYPE_NAMES        # flat {object_type_id: name}
    object_types.EXTENSION_BASE           # object_type IDs >= 32 are user extensions

Shape note
----------

Conditions / actions / expressions use a *nested* `{object_type_id: {code: name}}`
map because the same `code` means different things depending on which object
type the event is attached to. Example from `CONDITION_SYSTEM_NAMES`:

    CONDITION_SYSTEM_NAMES[2][-81]   == "ObjectClicked"
    CONDITION_SYSTEM_NAMES[9][-81]   == "SubApplicationFrameChanged"

Flattening these would collide. The name-resolution layer consumes the
`(code, object_type_id)` pair; callers must preserve both keys.

`PARAMETER_NAMES` and `OBJECT_TYPE_NAMES` are genuinely flat in the source
and stay flat here.

These modules are data-only — no behaviour, no imports beyond the module
docstring. They're ground truth for the name-resolver layer; tests
spot-check a few well-known entries per table rather than pinning every
row, because the tables themselves are the oracle.
"""

from __future__ import annotations

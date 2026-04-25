//! Runtime state shelves.
//!
//! The Rust mini-Clickteam engine stores all gameplay state as flat
//! shelves rather than semantic gameplay structs. This is deliberate:
//! the original FNAF event rows address numbered/named shelves
//! (counters by name, alterables by slot index, group flags by group
//! id) and a runtime that reshapes them into "Bonnie", "Freddy",
//! "Camera" structs would diverge from the source-of-truth event
//! rows the moment FNAF reuses a slot for an unrelated purpose.
//!
//! See the tree node *Runtime State Shelves* for the full shelf
//! catalog. Slices land here one shelf at a time:
//!
//! 1. [`counters`] — global counter shelf (this slice).
//! 2. Per-Active alterables (26 numbered i32 slots per instance) — TODO.
//! 3. Per-Active scalars (x/y/visibility/animation/...) — TODO.
//! 4. Per-event-group flags (Once/Every/NotAlways) — TODO.
//! 5. Sound channels, pending frame transition, input — TODO.

pub mod active_scalars;
pub mod alterables;
pub mod counters;
pub mod frame_transition;
pub mod group_flags;
pub mod sound_channels;

pub use active_scalars::{ActiveScalars, ActiveState};
pub use alterables::{Alterables, InstanceId, SLOT_COUNT};
pub use counters::Counters;
pub use frame_transition::{FrameId, FrameTransition, PendingFrameTransition};
pub use group_flags::{GroupFlags, GroupId, GroupState};
pub use sound_channels::{ChannelId, ChannelState, SoundChannels, MAX_VOLUME};

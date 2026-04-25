//! Per-event-group flag shelf.
//!
//! Clickteam Fusion event rows can be tagged with three rate-limiting
//! constructs that need persistent state between engine ticks:
//!
//! * **Once** — the event fires at most one time per "lifetime"
//!   (typically: from first true-condition until a frame transition
//!   resets the shelf). Stored as `fired: bool`.
//! * **Every N** — the event fires once per N ticks. Stored as
//!   `last_tick: u32`, the engine tick number on which the row most
//!   recently fired; the engine compares `current_tick - last_tick`
//!   against N at evaluation time.
//! * **NotAlways** — the event fires only on the rising edge of its
//!   conditions; once fired, it inhibits until conditions go false
//!   again. Stored as `inhibit_until_reset: bool`.
//!
//! These three constructs are independent; a single event group may be
//! tagged with none, one, or (less commonly) more than one of them.
//! This shelf stores all three fields per group with safe defaults so
//! groups that don't use a given construct simply leave the
//! corresponding field at its zero/false default.
//!
//! Engine ownership boundary: this shelf is dumb storage. Deciding
//! when `fired` should be set, when `last_tick` should advance, or
//! when `inhibit_until_reset` should clear is the job of the
//! op-execution layer, not this module. See the tree node
//! *Hard Op Semantics* for the conformance contract.
//!
//! Frame transitions clear the entire shelf via [`GroupFlags::clear`].
//! Within a frame, individual groups may also be reset (for example,
//! when a NotAlways group's conditions go false) via
//! [`GroupFlags::reset_not_always`].

use std::collections::HashMap;

/// Stable identifier for an event group from the runtime pack. Matches
/// the parser-side event-group id width.
pub type GroupId = u32;

/// Per-group rate-limiting flags. Defaults are: `fired=false` (Once
/// has not fired yet), `last_tick=0` (Every is "fresh" — engine must
/// special-case the never-fired case at op-execution time), and
/// `inhibit_until_reset=false` (NotAlways is armed).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GroupState {
    pub fired: bool,
    pub last_tick: u32,
    pub inhibit_until_reset: bool,
}

/// Per-group flag shelf, sparse over groups.
#[derive(Debug, Default, Clone)]
pub struct GroupFlags {
    groups: HashMap<GroupId, GroupState>,
}

impl GroupFlags {
    /// Create an empty group-flag shelf.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a group's flags. Unknown groups return the all-default
    /// record, which is the same record an unfired/uninhibited group
    /// would have anyway.
    pub fn get(&self, group_id: GroupId) -> GroupState {
        self.groups.get(&group_id).copied().unwrap_or_default()
    }

    /// Mutable accessor that auto-vivifies a group with default flags.
    /// Used by the engine to mark `fired`, advance `last_tick`, or
    /// raise `inhibit_until_reset` after a row has been evaluated.
    pub fn entry(&mut self, group_id: GroupId) -> &mut GroupState {
        self.groups.entry(group_id).or_default()
    }

    /// Borrowed accessor for trace snapshots; `None` if the group has
    /// never been written.
    pub fn snapshot(&self, group_id: GroupId) -> Option<&GroupState> {
        self.groups.get(&group_id)
    }

    /// Clear NotAlways inhibition on a single group when its
    /// conditions go false. Convenience for the engine's NotAlways
    /// rising-edge logic; equivalent to `entry(g).inhibit_until_reset
    /// = false` but auto-removes the group entirely if the cleared
    /// state is otherwise default, to keep the table small.
    pub fn reset_not_always(&mut self, group_id: GroupId) {
        if let Some(state) = self.groups.get_mut(&group_id) {
            state.inhibit_until_reset = false;
            if *state == GroupState::default() {
                self.groups.remove(&group_id);
            }
        }
    }

    /// Clear every group flag. Called by the engine on frame
    /// transitions, where Once/Every/NotAlways state resets.
    pub fn clear(&mut self) {
        self.groups.clear();
    }

    /// Iterate every known group id. Order is unspecified.
    pub fn groups(&self) -> impl Iterator<Item = GroupId> + '_ {
        self.groups.keys().copied()
    }

    /// Number of groups with non-default flags.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    /// True iff every group is at default flags.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_group_returns_default() {
        let g = GroupFlags::new();
        let state = g.get(42);
        assert_eq!(state, GroupState::default());
        assert!(!state.fired);
        assert_eq!(state.last_tick, 0);
        assert!(!state.inhibit_until_reset);
        assert!(g.is_empty());
    }

    #[test]
    fn entry_creates_with_default_and_persists_writes() {
        let mut g = GroupFlags::new();
        {
            let s = g.entry(7);
            assert_eq!(*s, GroupState::default());
            s.fired = true;
            s.last_tick = 100;
            s.inhibit_until_reset = true;
        }
        let got = g.get(7);
        assert!(got.fired);
        assert_eq!(got.last_tick, 100);
        assert!(got.inhibit_until_reset);
    }

    #[test]
    fn fired_persists_across_reads() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        for _ in 0..5 {
            assert!(g.get(1).fired);
        }
    }

    #[test]
    fn last_tick_persists_and_advances() {
        let mut g = GroupFlags::new();
        g.entry(1).last_tick = 30;
        assert_eq!(g.get(1).last_tick, 30);
        g.entry(1).last_tick = 60;
        assert_eq!(g.get(1).last_tick, 60);
    }

    #[test]
    fn inhibit_persists_and_clears() {
        let mut g = GroupFlags::new();
        g.entry(1).inhibit_until_reset = true;
        assert!(g.get(1).inhibit_until_reset);
        g.entry(1).inhibit_until_reset = false;
        assert!(!g.get(1).inhibit_until_reset);
    }

    #[test]
    fn groups_are_independent() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        g.entry(2).last_tick = 50;
        g.entry(3).inhibit_until_reset = true;
        assert!(g.get(1).fired);
        assert!(!g.get(1).inhibit_until_reset);
        assert_eq!(g.get(2).last_tick, 50);
        assert!(!g.get(2).fired);
        assert!(g.get(3).inhibit_until_reset);
        assert_eq!(g.get(3).last_tick, 0);
        assert_eq!(g.get(99), GroupState::default());
    }

    #[test]
    fn fields_within_group_independent() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        let s = g.get(1);
        assert!(s.fired);
        assert!(!s.inhibit_until_reset);
        assert_eq!(s.last_tick, 0);
    }

    #[test]
    fn reset_not_always_clears_inhibit_only() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        g.entry(1).inhibit_until_reset = true;
        g.entry(1).last_tick = 50;
        g.reset_not_always(1);
        let s = g.get(1);
        assert!(s.fired); // Once flag untouched
        assert_eq!(s.last_tick, 50); // Every tick untouched
        assert!(!s.inhibit_until_reset); // NotAlways cleared
    }

    #[test]
    fn reset_not_always_removes_default_group() {
        // If the only thing the group had was inhibit_until_reset and
        // it's been cleared, the group should be GC'd from the table
        // to keep the shelf compact.
        let mut g = GroupFlags::new();
        g.entry(7).inhibit_until_reset = true;
        assert_eq!(g.len(), 1);
        g.reset_not_always(7);
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn reset_not_always_keeps_group_with_other_flags() {
        let mut g = GroupFlags::new();
        g.entry(7).fired = true;
        g.entry(7).inhibit_until_reset = true;
        g.reset_not_always(7);
        // Group should still exist because `fired` is set.
        assert_eq!(g.len(), 1);
        assert!(g.get(7).fired);
        assert!(!g.get(7).inhibit_until_reset);
    }

    #[test]
    fn reset_not_always_on_unknown_group_is_noop() {
        let mut g = GroupFlags::new();
        g.reset_not_always(123);
        assert!(g.is_empty());
    }

    #[test]
    fn clear_resets_every_group() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        g.entry(2).last_tick = 100;
        g.entry(3).inhibit_until_reset = true;
        assert_eq!(g.len(), 3);
        g.clear();
        assert!(g.is_empty());
        assert_eq!(g.get(1), GroupState::default());
        assert_eq!(g.get(2), GroupState::default());
        assert_eq!(g.get(3), GroupState::default());
    }

    #[test]
    fn snapshot_returns_state_or_none() {
        let mut g = GroupFlags::new();
        g.entry(42).fired = true;
        let snap = g.snapshot(42).expect("group should exist");
        assert!(snap.fired);
        assert!(g.snapshot(99).is_none());
    }

    #[test]
    fn groups_iter_yields_only_known_ids() {
        let mut g = GroupFlags::new();
        g.entry(1).fired = true;
        g.entry(2).fired = true;
        g.entry(99).fired = true;
        let mut ids: Vec<GroupId> = g.groups().collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 99]);
    }

    #[test]
    fn last_tick_supports_full_u32_range() {
        let mut g = GroupFlags::new();
        g.entry(1).last_tick = u32::MAX;
        assert_eq!(g.get(1).last_tick, u32::MAX);
    }

    #[test]
    fn entry_does_not_clobber_existing_state() {
        let mut g = GroupFlags::new();
        g.entry(5).fired = true;
        let s = g.entry(5);
        assert!(s.fired);
        s.last_tick = 99;
        assert!(g.get(5).fired);
        assert_eq!(g.get(5).last_tick, 99);
    }
}

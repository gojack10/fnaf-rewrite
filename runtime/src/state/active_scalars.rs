//! Per-Active scalar shelf.
//!
//! Where [`crate::state::alterables`] holds the 26 numbered i32 slots
//! every Active instance addresses by index, this shelf holds the
//! fixed-name engine-side scalars: position, visibility, and animation
//! state. These fields are read and written by Clickteam ops with
//! dedicated names (SetPosition, SetVisible, ChangeAnimation, ...)
//! rather than through a numeric slot index, which is why they live in
//! a struct rather than a Vec.
//!
//! Fields included in this slice are the engine-relevant scalars only:
//!
//! * `x`, `y` — instance position in frame coordinates.
//! * `visible` — whether the instance is currently shown.
//! * `animation_id` — current animation handle from ObjectCommon.
//! * `animation_frame` — frame index inside the current animation.
//! * `animation_finished` — set by the engine when a non-looping
//!   animation completes; consumed by event-row conditions like
//!   "Animation X has finished".
//!
//! Renderer-only fields (z/layer, density, alpha) are intentionally
//! deferred until the renderer pass starts and we know which shelf
//! actually owns them — those may belong on the renderer side rather
//! than the engine side, and committing to a layout now would risk
//! locking in the wrong owner.
//!
//! Contract:
//! * Reading an unknown instance returns `ActiveState::default()`
//!   (all-zero numerics, `visible=false`, both bools false).
//! * `entry` auto-vivifies an instance with the same default and
//!   returns a mutable reference for in-place updates.
//! * `destroy` removes an instance entirely (for Clickteam Destroy).
//! * The engine's Spawn op is responsible for the Clickteam default
//!   `visible=true` on creation; the shelf does not infer it.

use std::collections::HashMap;

use crate::state::alterables::InstanceId;

/// Engine-side scalar state for a single Active instance.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ActiveState {
    pub x: i32,
    pub y: i32,
    pub visible: bool,
    pub animation_id: u16,
    pub animation_frame: u16,
    pub animation_finished: bool,
}

/// Per-instance fixed-name scalar shelf.
#[derive(Debug, Default, Clone)]
pub struct ActiveScalars {
    instances: HashMap<InstanceId, ActiveState>,
}

impl ActiveScalars {
    /// Create an empty scalar shelf with no live instances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read an instance's scalars. Unknown instances return the
    /// all-default record (caller can interpret default `visible=false`
    /// as "no such instance" because the Spawn op will explicitly set
    /// visibility on creation).
    pub fn get(&self, instance_id: InstanceId) -> ActiveState {
        self.instances.get(&instance_id).copied().unwrap_or_default()
    }

    /// Mutable accessor that auto-vivifies an instance with default
    /// scalars on first write. Used by the engine's setter ops.
    pub fn entry(&mut self, instance_id: InstanceId) -> &mut ActiveState {
        self.instances.entry(instance_id).or_default()
    }

    /// Borrowed accessor for trace snapshots; `None` if the instance
    /// is not known.
    pub fn snapshot(&self, instance_id: InstanceId) -> Option<&ActiveState> {
        self.instances.get(&instance_id)
    }

    /// Remove an instance entirely. Returns whether the instance was
    /// known before the call.
    pub fn destroy(&mut self, instance_id: InstanceId) -> bool {
        self.instances.remove(&instance_id).is_some()
    }

    /// Iterate every known instance id. Order is unspecified.
    pub fn instances(&self) -> impl Iterator<Item = InstanceId> + '_ {
        self.instances.keys().copied()
    }

    /// Number of live instances on the shelf.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// True iff no instance has ever been written.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_instance_returns_default() {
        let s = ActiveScalars::new();
        let got = s.get(42);
        assert_eq!(got, ActiveState::default());
        assert_eq!(got.x, 0);
        assert_eq!(got.y, 0);
        assert!(!got.visible);
        assert_eq!(got.animation_id, 0);
        assert_eq!(got.animation_frame, 0);
        assert!(!got.animation_finished);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn entry_creates_with_default_and_persists_writes() {
        let mut s = ActiveScalars::new();
        {
            let st = s.entry(7);
            assert_eq!(*st, ActiveState::default());
            st.x = 100;
            st.y = -50;
            st.visible = true;
        }
        let got = s.get(7);
        assert_eq!(got.x, 100);
        assert_eq!(got.y, -50);
        assert!(got.visible);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fields_within_an_instance_are_independent() {
        let mut s = ActiveScalars::new();
        let st = s.entry(1);
        st.x = 10;
        st.y = 20;
        st.visible = true;
        st.animation_id = 5;
        st.animation_frame = 3;
        st.animation_finished = true;
        let got = s.get(1);
        assert_eq!(got.x, 10);
        assert_eq!(got.y, 20);
        assert!(got.visible);
        assert_eq!(got.animation_id, 5);
        assert_eq!(got.animation_frame, 3);
        assert!(got.animation_finished);
    }

    #[test]
    fn instances_are_independent() {
        let mut s = ActiveScalars::new();
        s.entry(1).x = 100;
        s.entry(2).x = 200;
        s.entry(3).x = 300;
        assert_eq!(s.get(1).x, 100);
        assert_eq!(s.get(2).x, 200);
        assert_eq!(s.get(3).x, 300);
        assert_eq!(s.get(4).x, 0); // unknown still defaults
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn partial_update_keeps_other_fields_at_default() {
        let mut s = ActiveScalars::new();
        s.entry(9).animation_id = 12;
        let got = s.get(9);
        assert_eq!(got.animation_id, 12);
        assert_eq!(got.x, 0);
        assert_eq!(got.y, 0);
        assert!(!got.visible);
        assert_eq!(got.animation_frame, 0);
        assert!(!got.animation_finished);
    }

    #[test]
    fn destroy_removes_instance() {
        let mut s = ActiveScalars::new();
        s.entry(11).x = 5;
        assert_eq!(s.get(11).x, 5);
        assert!(s.destroy(11));
        assert_eq!(s.get(11), ActiveState::default());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn destroy_returns_false_for_unknown_instance() {
        let mut s = ActiveScalars::new();
        assert!(!s.destroy(123));
        s.entry(1).x = 1;
        assert!(!s.destroy(2));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn destroy_does_not_leak_to_other_instances() {
        let mut s = ActiveScalars::new();
        s.entry(1).x = 100;
        s.entry(2).x = 200;
        s.destroy(1);
        assert_eq!(s.get(1).x, 0);
        assert_eq!(s.get(2).x, 200);
    }

    #[test]
    fn snapshot_returns_full_state_or_none() {
        let mut s = ActiveScalars::new();
        s.entry(42).animation_id = 7;
        s.entry(42).visible = true;
        let snap = s.snapshot(42).expect("instance should exist");
        assert!(snap.visible);
        assert_eq!(snap.animation_id, 7);
        assert!(s.snapshot(99).is_none());
    }

    #[test]
    fn instances_iter_yields_only_known_ids() {
        let mut s = ActiveScalars::new();
        s.entry(1);
        s.entry(2);
        s.entry(99);
        let mut ids: Vec<InstanceId> = s.instances().collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 99]);
    }

    #[test]
    fn position_supports_negative_and_extreme_values() {
        let mut s = ActiveScalars::new();
        let st = s.entry(1);
        st.x = i32::MIN;
        st.y = i32::MAX;
        let got = s.get(1);
        assert_eq!(got.x, i32::MIN);
        assert_eq!(got.y, i32::MAX);
    }

    #[test]
    fn animation_handles_full_u16_range() {
        let mut s = ActiveScalars::new();
        let st = s.entry(1);
        st.animation_id = u16::MAX;
        st.animation_frame = u16::MAX;
        let got = s.get(1);
        assert_eq!(got.animation_id, u16::MAX);
        assert_eq!(got.animation_frame, u16::MAX);
    }

    #[test]
    fn entry_returns_existing_instance_without_clobbering() {
        let mut s = ActiveScalars::new();
        s.entry(5).x = 42;
        // Second entry call must NOT reset to default; it must return
        // the same record so the engine can read-modify-write safely.
        let st = s.entry(5);
        assert_eq!(st.x, 42);
        st.y = 99;
        assert_eq!(s.get(5).x, 42);
        assert_eq!(s.get(5).y, 99);
    }
}

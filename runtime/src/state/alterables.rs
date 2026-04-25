//! Per-Active alterable shelf.
//!
//! Every Clickteam Active instance carries a fixed-size array of
//! `i32` "alterables" addressed by slot index. Vanilla Clickteam Fusion
//! 2.5 exposes 26 slots per instance (extended builds add 26 more, but
//! the parser-side ObjectCommon Decoder probe confirmed FNAF 1 stays
//! inside the vanilla range). Within those 26 slots, FNAF 1 only ever
//! reads or writes slots 0-18 — see the warning on the
//! ObjectCommon Decoder node.
//!
//! Per the same probe, Counter init values are NOT in ObjectCommon —
//! all 43 baseline 180-byte Counter property bodies carry byte-identical
//! bytes 60-96, so per-Active alterables similarly default to 0 on
//! instance spawn rather than coming from the runtime pack. Game rules
//! are responsible for explicit init via SetAlterableValue ops on
//! StartOfFrame; this shelf only stores values written by ops.
//!
//! Contract:
//! * Reading a slot of an unknown instance returns 0.
//! * Reading an unwritten slot of a known instance returns 0.
//! * `set` and `add` auto-vivify a known instance with a zeroed array.
//! * `destroy` removes an instance entirely (used when a Clickteam
//!   Destroy action fires).
//! * Slot indices outside `0..SLOT_COUNT` panic via array bounds —
//!   that is a programming error / schema-drift signal, not a runtime
//!   condition the engine can recover from.

use std::collections::HashMap;

/// Number of alterable slots per Active instance in vanilla Clickteam
/// Fusion 2.5. FNAF 1 stays inside this range and only touches slots
/// 0-18 in practice.
pub const SLOT_COUNT: usize = 26;

/// Stable identifier for a single Active instance at runtime. Matches
/// the `handle` width Clickteam uses internally and the parser exposes
/// via FrameItemInstance.
pub type InstanceId = u32;

/// Per-instance alterable storage. Internally `HashMap<InstanceId,
/// [i32; SLOT_COUNT]>` — sparse over instances, dense over slots,
/// because every instance touches at least slot 0 in practice but a
/// frame may legitimately have many fewer Active instances live than
/// the parser-time ObjectInfo count.
#[derive(Debug, Default, Clone)]
pub struct Alterables {
    instances: HashMap<InstanceId, [i32; SLOT_COUNT]>,
}

impl Alterables {
    /// Create an empty alterable shelf with no live instances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a slot. Unknown instance or unwritten slot both return 0.
    /// Panics if `slot >= SLOT_COUNT`.
    pub fn get(&self, instance_id: InstanceId, slot: usize) -> i32 {
        match self.instances.get(&instance_id) {
            Some(slots) => slots[slot],
            None => {
                // Bounds-check the slot even on the miss path so a
                // bogus slot index surfaces consistently regardless of
                // whether the instance exists yet.
                assert!(
                    slot < SLOT_COUNT,
                    "alterable slot {slot} out of range (max {})",
                    SLOT_COUNT - 1
                );
                0
            }
        }
    }

    /// Write a slot, auto-vivifying the instance with a zeroed array
    /// if it is not already known. Panics if `slot >= SLOT_COUNT`.
    pub fn set(&mut self, instance_id: InstanceId, slot: usize, value: i32) {
        let slots = self.instances.entry(instance_id).or_insert([0; SLOT_COUNT]);
        slots[slot] = value;
    }

    /// Add `delta` to a slot (creating the instance and the slot at 0
    /// if absent) and return the new value. Wraps on i32 overflow,
    /// matching the Clickteam 32-bit signed semantics already
    /// established for the Counters shelf.
    pub fn add(&mut self, instance_id: InstanceId, slot: usize, delta: i32) -> i32 {
        let slots = self.instances.entry(instance_id).or_insert([0; SLOT_COUNT]);
        slots[slot] = slots[slot].wrapping_add(delta);
        slots[slot]
    }

    /// Remove an instance entirely. Returns whether the instance was
    /// known before the call. Used by the runtime's Destroy op.
    pub fn destroy(&mut self, instance_id: InstanceId) -> bool {
        self.instances.remove(&instance_id).is_some()
    }

    /// Full slot array for an instance, or `None` if the instance is
    /// not known. Useful for trace snapshots; not for hot paths.
    pub fn snapshot(&self, instance_id: InstanceId) -> Option<&[i32; SLOT_COUNT]> {
        self.instances.get(&instance_id)
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
    fn missing_instance_reads_zero() {
        let a = Alterables::new();
        assert_eq!(a.get(0, 0), 0);
        assert_eq!(a.get(99, 5), 0);
        assert_eq!(a.get(u32::MAX, 25), 0);
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn set_creates_instance_and_zeros_other_slots() {
        let mut a = Alterables::new();
        a.set(7, 3, 42);
        assert_eq!(a.get(7, 3), 42);
        // Other slots on the same instance remain zero.
        for s in 0..SLOT_COUNT {
            if s != 3 {
                assert_eq!(a.get(7, s), 0, "slot {s} should be zero");
            }
        }
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn set_then_get_roundtrips_per_slot() {
        let mut a = Alterables::new();
        a.set(1, 0, 10);
        a.set(1, 18, 99);
        a.set(1, 25, -7);
        assert_eq!(a.get(1, 0), 10);
        assert_eq!(a.get(1, 18), 99);
        assert_eq!(a.get(1, 25), -7);
    }

    #[test]
    fn add_creates_then_accumulates() {
        let mut a = Alterables::new();
        assert_eq!(a.add(2, 0, 5), 5);
        assert_eq!(a.add(2, 0, 3), 8);
        assert_eq!(a.add(2, 0, -10), -2);
        assert_eq!(a.get(2, 0), -2);
    }

    #[test]
    fn add_after_set_starts_from_set_value() {
        let mut a = Alterables::new();
        a.set(3, 4, 100);
        assert_eq!(a.add(3, 4, 1), 101);
        assert_eq!(a.add(3, 4, -50), 51);
    }

    #[test]
    fn instances_are_independent() {
        let mut a = Alterables::new();
        a.set(10, 0, 1);
        a.set(20, 0, 2);
        a.set(30, 0, 3);
        assert_eq!(a.get(10, 0), 1);
        assert_eq!(a.get(20, 0), 2);
        assert_eq!(a.get(30, 0), 3);
        assert_eq!(a.get(40, 0), 0);
        assert_eq!(a.len(), 3);
    }

    #[test]
    fn slots_within_an_instance_are_independent() {
        let mut a = Alterables::new();
        for s in 0..SLOT_COUNT {
            a.set(5, s, s as i32 * 10);
        }
        for s in 0..SLOT_COUNT {
            assert_eq!(a.get(5, s), s as i32 * 10);
        }
    }

    #[test]
    fn destroy_removes_instance() {
        let mut a = Alterables::new();
        a.set(11, 7, 123);
        assert_eq!(a.get(11, 7), 123);
        assert!(a.destroy(11));
        assert_eq!(a.get(11, 7), 0);
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn destroy_returns_false_for_unknown_instance() {
        let mut a = Alterables::new();
        assert!(!a.destroy(999));
        a.set(1, 0, 1);
        assert!(!a.destroy(2));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn destroy_does_not_leak_to_other_instances() {
        let mut a = Alterables::new();
        a.set(1, 0, 100);
        a.set(2, 0, 200);
        a.destroy(1);
        assert_eq!(a.get(1, 0), 0);
        assert_eq!(a.get(2, 0), 200);
    }

    #[test]
    fn snapshot_returns_full_array() {
        let mut a = Alterables::new();
        a.set(42, 0, 1);
        a.set(42, 18, 18);
        let snap = a.snapshot(42).expect("instance should exist");
        assert_eq!(snap[0], 1);
        assert_eq!(snap[18], 18);
        // Untouched slots are zero.
        assert_eq!(snap[1], 0);
        assert_eq!(snap[25], 0);
        assert_eq!(snap.len(), SLOT_COUNT);
    }

    #[test]
    fn snapshot_returns_none_for_unknown_instance() {
        let a = Alterables::new();
        assert!(a.snapshot(7).is_none());
    }

    #[test]
    fn instances_iter_yields_only_known_ids() {
        let mut a = Alterables::new();
        a.set(1, 0, 0);
        a.set(2, 0, 0);
        a.set(99, 0, 0);
        let mut ids: Vec<InstanceId> = a.instances().collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 99]);
    }

    #[test]
    fn add_wraps_on_overflow_per_slot() {
        let mut a = Alterables::new();
        a.set(1, 0, i32::MAX);
        assert_eq!(a.add(1, 0, 1), i32::MIN);
        a.set(1, 1, i32::MIN);
        assert_eq!(a.add(1, 1, -1), i32::MAX);
    }

    #[test]
    #[should_panic]
    fn slot_out_of_range_panics_on_set() {
        let mut a = Alterables::new();
        a.set(1, SLOT_COUNT, 0);
    }

    #[test]
    #[should_panic]
    fn slot_out_of_range_panics_on_get_known_instance() {
        let mut a = Alterables::new();
        a.set(1, 0, 0); // instance is now known
        let _ = a.get(1, SLOT_COUNT);
    }

    #[test]
    #[should_panic]
    fn slot_out_of_range_panics_on_get_unknown_instance() {
        let a = Alterables::new();
        let _ = a.get(1, SLOT_COUNT);
    }

    #[test]
    fn fnaf_active_slot_range_is_supported() {
        // Anchor test documenting the FNAF 1 evidence: events touch
        // slots 0-18. All of these must roundtrip without surprise.
        let mut a = Alterables::new();
        for s in 0..=18usize {
            a.set(123, s, s as i32);
        }
        for s in 0..=18usize {
            assert_eq!(a.get(123, s), s as i32);
        }
    }
}

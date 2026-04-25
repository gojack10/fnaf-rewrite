//! Global counter shelf.
//!
//! Clickteam's "Counter" object_type=7 plus FNAF's named global counter
//! variables collapse, at runtime, into a flat name-keyed `i32` table. The
//! parser-side runtime pack records both static Counter ObjectInfos
//! (44 in FNAF 1) and the named globals referenced by event rows
//! (`viewing`, `power left`, `time of day`, `night number`, `move who?`,
//! ...). At engine startup all observed names default to 0; FNAF does
//! explicit `SetCounterValue` initialization in 6 StartOfFrame actions.
//!
//! This shelf is intentionally dumb. It does not know what a counter
//! "means". Game rules live in the event rows the runtime executes
//! against this shelf.
//!
//! Contract:
//! * Reading a name that has never been written returns 0 (matches the
//!   Clickteam GetValue default and the implicit-zero behavior FNAF
//!   relies on for fresh counters).
//! * `set` overwrites unconditionally.
//! * `add` accumulates and returns the new value.

use std::collections::HashMap;

/// Flat name-keyed `i32` shelf for global counters and named counter
/// variables.
#[derive(Debug, Default, Clone)]
pub struct Counters {
    values: HashMap<String, i32>,
}

impl Counters {
    /// Create an empty counter shelf. All reads will default to 0 until
    /// something is written.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a counter by name. Missing names return 0 by contract.
    pub fn get(&self, name: &str) -> i32 {
        self.values.get(name).copied().unwrap_or(0)
    }

    /// Overwrite a counter unconditionally.
    pub fn set(&mut self, name: &str, value: i32) {
        self.values.insert(name.to_owned(), value);
    }

    /// Add `delta` to a counter (creating it at 0 first if absent) and
    /// return the new value. `delta` may be negative; this is also how
    /// `SubtractFromCounter` lowers.
    pub fn add(&mut self, name: &str, delta: i32) -> i32 {
        let entry = self.values.entry(name.to_owned()).or_insert(0);
        *entry = entry.wrapping_add(delta);
        *entry
    }

    /// Number of distinct counter names that have ever been written.
    /// A counter that was set then later read still counts; a name that
    /// has only been read does not.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True iff no counter has ever been written.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate every written `(name, value)` pair. Order is unspecified.
    /// Useful for snapshots / trace dumps; not for hot paths.
    pub fn iter(&self) -> impl Iterator<Item = (&str, i32)> {
        self.values.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_name_reads_as_zero() {
        let c = Counters::new();
        assert_eq!(c.get("viewing"), 0);
        assert_eq!(c.get("power left"), 0);
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn set_then_get_roundtrips() {
        let mut c = Counters::new();
        c.set("night number", 1);
        assert_eq!(c.get("night number"), 1);
        c.set("night number", 4);
        assert_eq!(c.get("night number"), 4);
    }

    #[test]
    fn set_supports_negative_and_zero() {
        let mut c = Counters::new();
        c.set("power left", -3);
        assert_eq!(c.get("power left"), -3);
        c.set("power left", 0);
        assert_eq!(c.get("power left"), 0);
        // Setting back to zero still keeps the name in the table; this
        // is fine — it just records that the engine has touched it.
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn add_creates_then_accumulates() {
        let mut c = Counters::new();
        // First add against a never-seen name starts from 0.
        assert_eq!(c.add("viewing", 1), 1);
        assert_eq!(c.add("viewing", 1), 2);
        assert_eq!(c.add("viewing", -1), 1);
        assert_eq!(c.get("viewing"), 1);
    }

    #[test]
    fn add_after_set_starts_from_set_value() {
        let mut c = Counters::new();
        c.set("time of day", 12);
        assert_eq!(c.add("time of day", 1), 13);
        assert_eq!(c.add("time of day", -2), 11);
    }

    #[test]
    fn names_are_independent() {
        let mut c = Counters::new();
        c.set("viewing", 5);
        c.add("power left", -1);
        c.set("night number", 1);
        assert_eq!(c.get("viewing"), 5);
        assert_eq!(c.get("power left"), -1);
        assert_eq!(c.get("night number"), 1);
        assert_eq!(c.get("nonexistent"), 0);
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn iter_yields_every_written_name() {
        let mut c = Counters::new();
        c.set("a", 1);
        c.set("b", 2);
        c.add("c", 3);
        let mut snapshot: Vec<(String, i32)> =
            c.iter().map(|(k, v)| (k.to_owned(), v)).collect();
        snapshot.sort();
        assert_eq!(
            snapshot,
            vec![
                ("a".to_owned(), 1),
                ("b".to_owned(), 2),
                ("c".to_owned(), 3),
            ]
        );
    }

    #[test]
    fn add_wraps_on_overflow() {
        // Clickteam counters are 32-bit signed; matching that contract
        // means we wrap on overflow rather than panic. FNAF should
        // never hit this in practice, but defining the behavior up
        // front keeps debug and release identical.
        let mut c = Counters::new();
        c.set("edge", i32::MAX);
        assert_eq!(c.add("edge", 1), i32::MIN);
        c.set("edge", i32::MIN);
        assert_eq!(c.add("edge", -1), i32::MAX);
    }

    #[test]
    fn default_matches_new() {
        let a: Counters = Counters::default();
        let b = Counters::new();
        assert!(a.is_empty());
        assert!(b.is_empty());
        assert_eq!(a.len(), b.len());
    }
}

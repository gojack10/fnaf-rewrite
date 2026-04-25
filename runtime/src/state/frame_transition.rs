//! Pending frame transition shelf.
//!
//! Clickteam events queue a frame change with `JumpToFrame(N)` /
//! `NextFrame` / `PreviousFrame` / `RestartFrame`. The engine does
//! NOT switch frames mid-tick — it finishes the current event-row
//! evaluation, then applies the queued frame change at the end of
//! the tick. This shelf models that single pending slot.
//!
//! Only one frame transition can be pending at a time. Multiple
//! `JumpToFrame` actions in the same tick collapse to the last one
//! issued, matching Clickteam Fusion's documented behavior.
//!
//! Engine ownership:
//! * Op execution layer calls `set_jump` / `set_restart` etc. when an
//!   action fires.
//! * The tick-end driver calls `take` to consume the pending slot and
//!   apply the transition; `take` clears the slot atomically.

/// Frame identifier from the runtime pack. Matches the parser-side
/// frame-handle width.
pub type FrameId = u32;

/// Kind of pending frame transition. Includes the four constructs
/// FNAF 1 actually issues; if the parser surfaces additional ones
/// later they get added here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameTransition {
    /// Explicit jump to a frame by id.
    Jump(FrameId),
    /// Step to the next frame in the runtime pack's frame order.
    Next,
    /// Step to the previous frame in the runtime pack's frame order.
    Previous,
    /// Restart the current frame from scratch.
    Restart,
}

/// Single-slot pending frame transition shelf.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PendingFrameTransition {
    pending: Option<FrameTransition>,
}

impl PendingFrameTransition {
    /// Create an empty pending-transition slot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the pending transition without consuming it. Useful for
    /// debug snapshots; engine should consume via `take`.
    pub fn peek(&self) -> Option<FrameTransition> {
        self.pending
    }

    /// True iff a transition is queued.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Queue an explicit jump to a frame. Replaces any prior queued
    /// transition (Clickteam collapse-to-last semantics).
    pub fn set_jump(&mut self, frame_id: FrameId) {
        self.pending = Some(FrameTransition::Jump(frame_id));
    }

    /// Queue a next-frame step. Replaces any prior queued transition.
    pub fn set_next(&mut self) {
        self.pending = Some(FrameTransition::Next);
    }

    /// Queue a previous-frame step. Replaces any prior queued transition.
    pub fn set_previous(&mut self) {
        self.pending = Some(FrameTransition::Previous);
    }

    /// Queue a frame restart. Replaces any prior queued transition.
    pub fn set_restart(&mut self) {
        self.pending = Some(FrameTransition::Restart);
    }

    /// Generic setter for callers that already have a `FrameTransition`
    /// value (e.g. dispatch from a parser-decoded action). Replaces
    /// any prior queued transition.
    pub fn set(&mut self, transition: FrameTransition) {
        self.pending = Some(transition);
    }

    /// Atomically read and clear the pending transition. Returns
    /// `None` if nothing was queued. Used by the tick-end driver.
    pub fn take(&mut self) -> Option<FrameTransition> {
        self.pending.take()
    }

    /// Clear any pending transition without consuming. Used when the
    /// engine wants to cancel a queued jump (rare).
    pub fn clear(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slot_is_not_pending() {
        let p = PendingFrameTransition::new();
        assert!(!p.is_pending());
        assert_eq!(p.peek(), None);
    }

    #[test]
    fn set_jump_makes_pending() {
        let mut p = PendingFrameTransition::new();
        p.set_jump(5);
        assert!(p.is_pending());
        assert_eq!(p.peek(), Some(FrameTransition::Jump(5)));
    }

    #[test]
    fn set_next_makes_pending() {
        let mut p = PendingFrameTransition::new();
        p.set_next();
        assert_eq!(p.peek(), Some(FrameTransition::Next));
    }

    #[test]
    fn set_previous_makes_pending() {
        let mut p = PendingFrameTransition::new();
        p.set_previous();
        assert_eq!(p.peek(), Some(FrameTransition::Previous));
    }

    #[test]
    fn set_restart_makes_pending() {
        let mut p = PendingFrameTransition::new();
        p.set_restart();
        assert_eq!(p.peek(), Some(FrameTransition::Restart));
    }

    #[test]
    fn generic_set_routes_through() {
        let mut p = PendingFrameTransition::new();
        p.set(FrameTransition::Jump(99));
        assert_eq!(p.peek(), Some(FrameTransition::Jump(99)));
    }

    #[test]
    fn later_set_replaces_earlier_in_same_tick() {
        // Clickteam collapses multiple transitions queued in one tick
        // to the last one issued. Confirm both same-kind and
        // different-kind replacement behaviors.
        let mut p = PendingFrameTransition::new();
        p.set_jump(1);
        p.set_jump(2);
        assert_eq!(p.peek(), Some(FrameTransition::Jump(2)));
        p.set_next();
        assert_eq!(p.peek(), Some(FrameTransition::Next));
        p.set_restart();
        assert_eq!(p.peek(), Some(FrameTransition::Restart));
    }

    #[test]
    fn take_consumes_and_returns_pending() {
        let mut p = PendingFrameTransition::new();
        p.set_jump(7);
        assert_eq!(p.take(), Some(FrameTransition::Jump(7)));
        assert!(!p.is_pending());
        assert_eq!(p.peek(), None);
    }

    #[test]
    fn take_returns_none_when_empty() {
        let mut p = PendingFrameTransition::new();
        assert_eq!(p.take(), None);
        assert!(!p.is_pending());
    }

    #[test]
    fn take_is_idempotent_after_consumption() {
        let mut p = PendingFrameTransition::new();
        p.set_jump(1);
        let first = p.take();
        let second = p.take();
        assert_eq!(first, Some(FrameTransition::Jump(1)));
        assert_eq!(second, None);
    }

    #[test]
    fn clear_drops_without_returning() {
        let mut p = PendingFrameTransition::new();
        p.set_next();
        p.clear();
        assert!(!p.is_pending());
        assert_eq!(p.peek(), None);
    }

    #[test]
    fn clear_is_safe_when_empty() {
        let mut p = PendingFrameTransition::new();
        p.clear();
        assert!(!p.is_pending());
    }

    #[test]
    fn jump_supports_full_frame_id_range() {
        let mut p = PendingFrameTransition::new();
        p.set_jump(0);
        assert_eq!(p.take(), Some(FrameTransition::Jump(0)));
        p.set_jump(u32::MAX);
        assert_eq!(p.take(), Some(FrameTransition::Jump(u32::MAX)));
    }
}

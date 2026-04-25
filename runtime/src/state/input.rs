//! Input shelf — keyboard and mouse state delivered by the host.
//!
//! The host event loop translates platform input events into calls on
//! these shelves; the engine reads them during op evaluation. Two
//! independent shelves are exposed:
//!
//! * [`KeyboardState`] — held keys plus per-tick press/release edges.
//! * [`MouseState`] — cursor position plus held-buttons and per-tick
//!   button press/release edges.
//!
//! Edge state (press/release this tick) lets event-row conditions like
//! "Upon pressing X" fire only on the rising edge. The engine MUST
//! call [`end_tick`](KeyboardState::end_tick) on both shelves at the
//! end of each tick to clear the edge sets — held state survives.
//!
//! Key codes use a `u32` alias to avoid committing to a specific
//! platform encoding here. The host translation layer is responsible
//! for picking a stable representation (typically Windows VK codes,
//! since the original FNAF runs on Win32).

use std::collections::HashSet;

/// Platform-agnostic key identifier. Host translation layer chooses
/// the encoding (Win32 VK codes are the natural choice for FNAF).
pub type KeyCode = u32;

/// Keyboard state: held keys plus this-tick edge sets.
#[derive(Debug, Default, Clone)]
pub struct KeyboardState {
    held: HashSet<KeyCode>,
    pressed: HashSet<KeyCode>,
    released: HashSet<KeyCode>,
}

impl KeyboardState {
    /// Create an empty keyboard state with no keys held.
    pub fn new() -> Self {
        Self::default()
    }

    /// Host-side: report a key-down event. Adds to held and to the
    /// pressed-this-tick edge set. Re-pressing an already-held key is
    /// idempotent and does NOT re-fire the edge.
    pub fn press(&mut self, key: KeyCode) {
        if self.held.insert(key) {
            self.pressed.insert(key);
        }
    }

    /// Host-side: report a key-up event. Removes from held and adds
    /// to the released-this-tick edge set. Releasing a key that was
    /// not held is idempotent and does NOT fire the edge.
    pub fn release(&mut self, key: KeyCode) {
        if self.held.remove(&key) {
            self.released.insert(key);
        }
    }

    /// Engine-side: true while the key is held.
    pub fn is_held(&self, key: KeyCode) -> bool {
        self.held.contains(&key)
    }

    /// Engine-side: true if the key transitioned to held this tick.
    pub fn was_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    /// Engine-side: true if the key transitioned to released this tick.
    pub fn was_released(&self, key: KeyCode) -> bool {
        self.released.contains(&key)
    }

    /// Iterate every currently-held key. Order is unspecified.
    pub fn held_keys(&self) -> impl Iterator<Item = KeyCode> + '_ {
        self.held.iter().copied()
    }

    /// Engine-side: clear the per-tick edge sets at end of tick. Held
    /// state survives.
    pub fn end_tick(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }

    /// Number of currently-held keys.
    pub fn len(&self) -> usize {
        self.held.len()
    }

    /// True iff no keys are held.
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }
}

/// Mouse buttons supported by the input shelf. Bit values match the
/// internal bitfield encoding so `as u8` round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MouseButton {
    Left = 0b001,
    Right = 0b010,
    Middle = 0b100,
}

/// Mouse state: cursor position plus held-buttons bitfield and edge
/// bitfields for press/release this tick.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MouseState {
    /// Cursor x in frame coordinates.
    pub x: i32,
    /// Cursor y in frame coordinates.
    pub y: i32,
    held: u8,
    pressed: u8,
    released: u8,
}

impl MouseState {
    /// Create a mouse state at origin with no buttons held.
    pub fn new() -> Self {
        Self::default()
    }

    /// Host-side: update cursor position.
    pub fn move_to(&mut self, x: i32, y: i32) {
        self.x = x;
        self.y = y;
    }

    /// Host-side: report a mouse-button-down event. Idempotent for
    /// already-held buttons (does NOT re-fire the press edge).
    pub fn press(&mut self, button: MouseButton) {
        let bit = button as u8;
        if (self.held & bit) == 0 {
            self.held |= bit;
            self.pressed |= bit;
        }
    }

    /// Host-side: report a mouse-button-up event. Idempotent for
    /// already-released buttons.
    pub fn release(&mut self, button: MouseButton) {
        let bit = button as u8;
        if (self.held & bit) != 0 {
            self.held &= !bit;
            self.released |= bit;
        }
    }

    /// Engine-side: true while a button is held.
    pub fn is_held(&self, button: MouseButton) -> bool {
        (self.held & button as u8) != 0
    }

    /// Engine-side: true if a button transitioned to held this tick.
    pub fn was_pressed(&self, button: MouseButton) -> bool {
        (self.pressed & button as u8) != 0
    }

    /// Engine-side: true if a button transitioned to released this tick.
    pub fn was_released(&self, button: MouseButton) -> bool {
        (self.released & button as u8) != 0
    }

    /// Engine-side: clear edge sets at end of tick. Held buttons and
    /// position survive.
    pub fn end_tick(&mut self) {
        self.pressed = 0;
        self.released = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Keyboard ---

    #[test]
    fn keyboard_starts_empty() {
        let k = KeyboardState::new();
        assert!(!k.is_held(0));
        assert!(!k.was_pressed(0));
        assert!(!k.was_released(0));
        assert!(k.is_empty());
    }

    #[test]
    fn key_press_sets_held_and_press_edge() {
        let mut k = KeyboardState::new();
        k.press(65);
        assert!(k.is_held(65));
        assert!(k.was_pressed(65));
        assert!(!k.was_released(65));
    }

    #[test]
    fn key_release_clears_held_and_sets_release_edge() {
        let mut k = KeyboardState::new();
        k.press(65);
        k.end_tick();
        k.release(65);
        assert!(!k.is_held(65));
        assert!(!k.was_pressed(65));
        assert!(k.was_released(65));
    }

    #[test]
    fn end_tick_clears_edges_but_not_held() {
        let mut k = KeyboardState::new();
        k.press(65);
        assert!(k.was_pressed(65));
        k.end_tick();
        assert!(!k.was_pressed(65));
        assert!(k.is_held(65));
    }

    #[test]
    fn double_press_does_not_refire_edge() {
        let mut k = KeyboardState::new();
        k.press(65);
        k.end_tick();
        k.press(65); // already held
        assert!(!k.was_pressed(65));
        assert!(k.is_held(65));
    }

    #[test]
    fn release_unheld_key_does_not_fire_edge() {
        let mut k = KeyboardState::new();
        k.release(99);
        assert!(!k.was_released(99));
        assert!(!k.is_held(99));
    }

    #[test]
    fn keys_are_independent() {
        let mut k = KeyboardState::new();
        k.press(65);
        k.press(66);
        k.press(67);
        assert!(k.is_held(65));
        assert!(k.is_held(66));
        assert!(k.is_held(67));
        assert!(!k.is_held(68));
        k.release(66);
        assert!(k.is_held(65));
        assert!(!k.is_held(66));
        assert!(k.is_held(67));
        assert_eq!(k.len(), 2);
    }

    #[test]
    fn held_keys_iter_yields_held_set() {
        let mut k = KeyboardState::new();
        k.press(1);
        k.press(2);
        k.press(99);
        let mut held: Vec<KeyCode> = k.held_keys().collect();
        held.sort();
        assert_eq!(held, vec![1, 2, 99]);
    }

    #[test]
    fn press_release_in_same_tick_yields_both_edges() {
        let mut k = KeyboardState::new();
        k.press(65);
        k.release(65);
        assert!(k.was_pressed(65));
        assert!(k.was_released(65));
        assert!(!k.is_held(65));
    }

    // --- Mouse ---

    #[test]
    fn mouse_starts_at_origin_with_no_buttons() {
        let m = MouseState::new();
        assert_eq!(m.x, 0);
        assert_eq!(m.y, 0);
        assert!(!m.is_held(MouseButton::Left));
        assert!(!m.is_held(MouseButton::Right));
        assert!(!m.is_held(MouseButton::Middle));
    }

    #[test]
    fn move_to_updates_position() {
        let mut m = MouseState::new();
        m.move_to(100, -50);
        assert_eq!(m.x, 100);
        assert_eq!(m.y, -50);
    }

    #[test]
    fn mouse_button_press_sets_held_and_press_edge() {
        let mut m = MouseState::new();
        m.press(MouseButton::Left);
        assert!(m.is_held(MouseButton::Left));
        assert!(m.was_pressed(MouseButton::Left));
        assert!(!m.was_released(MouseButton::Left));
    }

    #[test]
    fn mouse_button_release_clears_held_and_sets_release_edge() {
        let mut m = MouseState::new();
        m.press(MouseButton::Right);
        m.end_tick();
        m.release(MouseButton::Right);
        assert!(!m.is_held(MouseButton::Right));
        assert!(m.was_released(MouseButton::Right));
    }

    #[test]
    fn mouse_buttons_are_independent_bits() {
        let mut m = MouseState::new();
        m.press(MouseButton::Left);
        m.press(MouseButton::Right);
        assert!(m.is_held(MouseButton::Left));
        assert!(m.is_held(MouseButton::Right));
        assert!(!m.is_held(MouseButton::Middle));
        m.release(MouseButton::Left);
        assert!(!m.is_held(MouseButton::Left));
        assert!(m.is_held(MouseButton::Right));
        assert!(m.was_released(MouseButton::Left));
        assert!(!m.was_released(MouseButton::Right));
    }

    #[test]
    fn mouse_double_press_does_not_refire_edge() {
        let mut m = MouseState::new();
        m.press(MouseButton::Left);
        m.end_tick();
        m.press(MouseButton::Left);
        assert!(!m.was_pressed(MouseButton::Left));
        assert!(m.is_held(MouseButton::Left));
    }

    #[test]
    fn mouse_release_unheld_does_not_fire_edge() {
        let mut m = MouseState::new();
        m.release(MouseButton::Middle);
        assert!(!m.was_released(MouseButton::Middle));
    }

    #[test]
    fn mouse_end_tick_clears_edges_but_not_held_or_position() {
        let mut m = MouseState::new();
        m.move_to(50, 50);
        m.press(MouseButton::Left);
        m.end_tick();
        assert_eq!(m.x, 50);
        assert_eq!(m.y, 50);
        assert!(m.is_held(MouseButton::Left));
        assert!(!m.was_pressed(MouseButton::Left));
    }

    #[test]
    fn mouse_position_supports_negative_and_extremes() {
        let mut m = MouseState::new();
        m.move_to(i32::MIN, i32::MAX);
        assert_eq!(m.x, i32::MIN);
        assert_eq!(m.y, i32::MAX);
    }

    #[test]
    fn mouse_press_release_same_tick_yields_both_edges() {
        let mut m = MouseState::new();
        m.press(MouseButton::Left);
        m.release(MouseButton::Left);
        assert!(m.was_pressed(MouseButton::Left));
        assert!(m.was_released(MouseButton::Left));
        assert!(!m.is_held(MouseButton::Left));
    }
}

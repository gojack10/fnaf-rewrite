//! Sound channel shelf.
//!
//! Clickteam Fusion's audio model addresses sound by numeric channel:
//! `PlaySoundOnChannel(N, sample, loops)`, `StopChannel(N)`,
//! `SetChannelVolume(N, vol)`, and conditions like
//! `IsChannelPlaying(N)` and `SampleNameOnChannel(N)`. Each channel
//! holds at most one currently-playing sample plus its loop and volume
//! parameters.
//!
//! This shelf models *engine-side* channel state — what the engine
//! believes is playing — not the actual audio backend. The dumb
//! renderer/audio layer consumes a queue of audio commands derived
//! from this state and plays them; if the backend stops a sample
//! early (channel preempted, sample ran out), the engine should be
//! told via a callback so it can clear the channel here. Keeping
//! engine and backend state separate matches the renderer/runtime
//! boundary established on [[Rendering Stack|7bff1fdf-1b94-468b-b054-08600666b811]]
//! and [[Rust Clickteam Runtime|707a434c-79d0-483b-8a07-8275dd3b770e]].
//!
//! Channel numbering is sparse (HashMap-backed) rather than a fixed
//! 32-channel array. Clickteam's vanilla limit is ~32 channels but
//! FNAF 1's actual usage has not been audited yet, and a sparse map
//! both costs less memory and surfaces "channel 99" usage as soon as
//! it appears rather than silently writing into a fixed slot.
//!
//! Volume range is 0-100 to match Clickteam's percentage scale; the
//! shelf clamps writes outside that range in the convenience `play`
//! helper but does NOT clamp on direct `entry` writes (callers using
//! `entry` are assumed to know what they're doing).

use std::collections::HashMap;

/// Stable identifier for an audio channel. Matches the parser-side
/// channel-number width.
pub type ChannelId = u32;

/// Maximum volume (Clickteam percentage scale, 0-100 inclusive).
pub const MAX_VOLUME: u8 = 100;

/// State of a single audio channel from the engine's point of view.
/// Default is silent (`sample_id=None`, not looping, volume=0).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChannelState {
    /// Sample currently assigned to this channel. `None` means the
    /// channel is silent. Sample id width matches the parser-side
    /// asset-bank id width (u32).
    pub sample_id: Option<u32>,
    /// True if the engine asked this channel to loop.
    pub looping: bool,
    /// Volume on the Clickteam 0-100 percentage scale.
    pub volume: u8,
}

impl ChannelState {
    /// True iff the channel is currently silent (no sample assigned).
    pub fn is_silent(&self) -> bool {
        self.sample_id.is_none()
    }
}

/// Sparse per-channel sound shelf.
#[derive(Debug, Default, Clone)]
pub struct SoundChannels {
    channels: HashMap<ChannelId, ChannelState>,
}

impl SoundChannels {
    /// Create an empty channel shelf with no allocated channels.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read a channel's state. Unknown channels return the silent
    /// default — same record an explicitly-stopped channel would have.
    pub fn get(&self, channel_id: ChannelId) -> ChannelState {
        self.channels.get(&channel_id).copied().unwrap_or_default()
    }

    /// Mutable accessor that auto-vivifies a channel with the silent
    /// default. Use this only when you're going to write multiple
    /// fields and want to skip the helper API.
    pub fn entry(&mut self, channel_id: ChannelId) -> &mut ChannelState {
        self.channels.entry(channel_id).or_default()
    }

    /// Borrowed accessor for trace snapshots; `None` if the channel
    /// has never been touched.
    pub fn snapshot(&self, channel_id: ChannelId) -> Option<&ChannelState> {
        self.channels.get(&channel_id)
    }

    /// Convenience: assign a sample to a channel. Volume is clamped
    /// into the Clickteam 0-100 percentage range.
    pub fn play(&mut self, channel_id: ChannelId, sample_id: u32, looping: bool, volume: u8) {
        let state = self.channels.entry(channel_id).or_default();
        state.sample_id = Some(sample_id);
        state.looping = looping;
        state.volume = volume.min(MAX_VOLUME);
    }

    /// Convenience: stop a channel. Returns whether the channel had a
    /// sample assigned before the call. The channel is GC'd from the
    /// table after stopping so the shelf stays compact.
    pub fn stop(&mut self, channel_id: ChannelId) -> bool {
        match self.channels.remove(&channel_id) {
            Some(state) => state.sample_id.is_some(),
            None => false,
        }
    }

    /// Stop every channel. Called by the engine on frame transitions
    /// where channel state should not bleed across frames.
    pub fn stop_all(&mut self) {
        self.channels.clear();
    }

    /// Iterate every known channel id. Order is unspecified.
    pub fn channels(&self) -> impl Iterator<Item = ChannelId> + '_ {
        self.channels.keys().copied()
    }

    /// Number of allocated channels (not necessarily playing — a
    /// channel that was touched via `entry` and left silent still
    /// counts).
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// True iff no channel has ever been allocated.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_channel_returns_silent_default() {
        let s = SoundChannels::new();
        let st = s.get(0);
        assert_eq!(st, ChannelState::default());
        assert!(st.is_silent());
        assert_eq!(st.sample_id, None);
        assert!(!st.looping);
        assert_eq!(st.volume, 0);
        assert!(s.is_empty());
    }

    #[test]
    fn entry_creates_with_silent_default() {
        let mut s = SoundChannels::new();
        let st = s.entry(7);
        assert!(st.is_silent());
        assert_eq!(st.volume, 0);
        assert!(!st.looping);
    }

    #[test]
    fn play_assigns_sample_loop_and_volume() {
        let mut s = SoundChannels::new();
        s.play(1, 42, true, 75);
        let st = s.get(1);
        assert_eq!(st.sample_id, Some(42));
        assert!(st.looping);
        assert_eq!(st.volume, 75);
        assert!(!st.is_silent());
    }

    #[test]
    fn play_clamps_volume_above_max() {
        let mut s = SoundChannels::new();
        s.play(1, 5, false, 200);
        assert_eq!(s.get(1).volume, MAX_VOLUME);
        s.play(1, 5, false, 100);
        assert_eq!(s.get(1).volume, 100);
        s.play(1, 5, false, 0);
        assert_eq!(s.get(1).volume, 0);
    }

    #[test]
    fn play_replaces_previous_sample() {
        let mut s = SoundChannels::new();
        s.play(1, 10, true, 50);
        s.play(1, 20, false, 80);
        let st = s.get(1);
        assert_eq!(st.sample_id, Some(20));
        assert!(!st.looping);
        assert_eq!(st.volume, 80);
    }

    #[test]
    fn stop_clears_and_gcs_channel() {
        let mut s = SoundChannels::new();
        s.play(3, 7, false, 50);
        assert_eq!(s.len(), 1);
        assert!(s.stop(3));
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        // Subsequent get returns silent default.
        assert!(s.get(3).is_silent());
    }

    #[test]
    fn stop_returns_false_for_unknown_channel() {
        let mut s = SoundChannels::new();
        assert!(!s.stop(99));
    }

    #[test]
    fn stop_returns_false_for_silent_channel() {
        let mut s = SoundChannels::new();
        // Touch via entry but never assign a sample.
        s.entry(1).volume = 50;
        assert!(!s.stop(1));
        // Channel is GC'd anyway.
        assert!(s.is_empty());
    }

    #[test]
    fn channels_are_independent() {
        let mut s = SoundChannels::new();
        s.play(1, 10, false, 50);
        s.play(2, 20, true, 75);
        s.play(3, 30, false, 100);
        let a = s.get(1);
        let b = s.get(2);
        let c = s.get(3);
        assert_eq!(a.sample_id, Some(10));
        assert_eq!(b.sample_id, Some(20));
        assert_eq!(c.sample_id, Some(30));
        assert!(!a.looping);
        assert!(b.looping);
        assert!(!c.looping);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn stop_does_not_leak_to_other_channels() {
        let mut s = SoundChannels::new();
        s.play(1, 10, false, 50);
        s.play(2, 20, false, 50);
        s.stop(1);
        assert!(s.get(1).is_silent());
        assert_eq!(s.get(2).sample_id, Some(20));
    }

    #[test]
    fn stop_all_clears_every_channel() {
        let mut s = SoundChannels::new();
        s.play(1, 1, true, 50);
        s.play(2, 2, false, 75);
        s.play(99, 3, true, 100);
        assert_eq!(s.len(), 3);
        s.stop_all();
        assert!(s.is_empty());
        assert!(s.get(1).is_silent());
        assert!(s.get(2).is_silent());
        assert!(s.get(99).is_silent());
    }

    #[test]
    fn snapshot_returns_state_or_none() {
        let mut s = SoundChannels::new();
        s.play(7, 42, true, 60);
        let snap = s.snapshot(7).expect("channel 7 should exist");
        assert_eq!(snap.sample_id, Some(42));
        assert!(snap.looping);
        assert_eq!(snap.volume, 60);
        assert!(s.snapshot(8).is_none());
    }

    #[test]
    fn channels_iter_yields_only_known_ids() {
        let mut s = SoundChannels::new();
        s.play(1, 1, false, 50);
        s.play(7, 2, false, 50);
        s.play(99, 3, false, 50);
        let mut ids: Vec<ChannelId> = s.channels().collect();
        ids.sort();
        assert_eq!(ids, vec![1, 7, 99]);
    }

    #[test]
    fn entry_does_not_clobber_existing_state() {
        let mut s = SoundChannels::new();
        s.play(1, 42, true, 75);
        let st = s.entry(1);
        assert_eq!(st.sample_id, Some(42));
        st.volume = 30;
        assert_eq!(s.get(1).sample_id, Some(42));
        assert_eq!(s.get(1).volume, 30);
        assert!(s.get(1).looping);
    }

    #[test]
    fn entry_writes_outside_play_helper_are_not_clamped() {
        // entry is the escape hatch; it does NOT clamp volume. play
        // is the safe convenience and DOES clamp. Document both.
        let mut s = SoundChannels::new();
        s.entry(1).volume = 200; // bypasses play()'s clamp
        assert_eq!(s.get(1).volume, 200);
    }

    #[test]
    fn channel_state_is_silent_predicate() {
        let mut s = SoundChannels::new();
        assert!(s.get(1).is_silent());
        s.play(1, 5, false, 50);
        assert!(!s.get(1).is_silent());
        s.stop(1);
        assert!(s.get(1).is_silent());
    }
}

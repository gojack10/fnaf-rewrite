//! Clickteam extension plugins implemented for FNAF 1.
//!
//! The 2026-04-24 runtime-usage probe (see
//! `parser/temp-probes/extension_runtime_usage_2026_04_24/`) showed that FNAF
//! 1 only touches two extension plugins out of three present in the runtime
//! pack:
//!
//! * `Date & Time` (object_type=34, handle 151) — 2 zero-arg expressions.
//! * `Ini`         (object_type=33, handle 10)  — 1 file-name action,
//!                                                 1 write action,
//!                                                 1 read expression.
//! * `Perspective` (object_type=32, handle 40)  — renderer-side only,
//!                                                 deferred until renderer
//!                                                 pass starts.
//!
//! Both extensions ignore their property bodies entirely at runtime — FNAF
//! either uses default values or overrides them via runtime actions. This is
//! why no parser-side body decoder is shipped for these types.

pub mod date_time;
pub mod ini;

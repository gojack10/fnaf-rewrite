//! Lean Rust mini-Clickteam runtime for FNAF 1.
//!
//! Current scope is the two Clickteam extension plugins FNAF actually uses.
//! Per the 2026-04-24 runtime-usage probe (parser-side), FNAF references
//! exactly three call surfaces across both extensions, with no body data
//! reads — see [`extensions::date_time`] and [`extensions::ini`] for the
//! authoritative call mapping and call-site evidence.
//!
//! The full mini-Clickteam engine (event loop, state shelves, op execution)
//! will be wired into this crate later; for now we ship the leaf-most
//! semantics first because they are independently testable and unblock the
//! engine work.

pub mod extensions;
pub mod state;

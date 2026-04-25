//! Date & Time extension (object_type=34, FNAF 1 handle 151).
//!
//! ## FNAF 1 usage
//!
//! Only two zero-argument expressions are referenced in the entire decoded
//! event corpus, both inside Frame 1 group 433 (StartOfFrame trigger):
//!
//! * `ExtExpression_85` → drives `SetCounterValue` on counter `day`.
//! * `ExtExpression_86` → drives `SetCounterValue` on counter `month`.
//!
//! No conditions, no actions, no other expressions. Property body is 510B
//! of display-widget config (Tahoma font name, RGB colors, alpha row); FNAF
//! never displays the widget so the body is dead weight at runtime.
//!
//! ## Antibodies
//!
//! Earlier tree state claimed these expressions drove the 12 AM → 6 AM
//! night clock. That was wrong — they read system *calendar* date, not
//! time-of-day. The night-clock progression is sourced elsewhere and is
//! still an open question for the engine work.
//!
//! See [[Date & Time Extension Runtime|60130def-b202-4fff-8793-bccbe629b2b2]]
//! on the FNAF Speedrun ideation tree for context.

use chrono::{Datelike, Local};

/// `Date & Time.ExtExpression_85` — system clock day-of-month, 1..=31.
///
/// Implemented as a free function (not a method on a `DateTime` struct)
/// because the FNAF call surface is two zero-arg expressions on a single
/// object instance — there is no per-instance state to carry. If FNAF 2+
/// needs configurable behavior (timezone, frozen-time-for-tests), this
/// becomes a method; for now, free function is the honest shape.
pub fn ext_expression_85() -> i32 {
    Local::now().day() as i32
}

/// `Date & Time.ExtExpression_86` — system clock month, 1..=12.
pub fn ext_expression_86() -> i32 {
    Local::now().month() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_in_valid_range() {
        let d = ext_expression_85();
        assert!(
            (1..=31).contains(&d),
            "ext_expression_85 returned {d}, expected 1..=31",
        );
    }

    #[test]
    fn month_in_valid_range() {
        let m = ext_expression_86();
        assert!(
            (1..=12).contains(&m),
            "ext_expression_86 returned {m}, expected 1..=12",
        );
    }

    #[test]
    fn day_matches_local_now() {
        // Two consecutive calls within a single test millisecond should agree.
        // This guards against future refactors that could accidentally route
        // through different time sources for the two expressions.
        let probe = Local::now();
        let d = ext_expression_85();
        let m = ext_expression_86();
        // Day/month can race across the midnight boundary if the test
        // straddles midnight by milliseconds — vanishingly rare but worth
        // tolerating one-tick drift to keep the test reliable on CI.
        assert!(
            d == probe.day() as i32 || (d - probe.day() as i32).abs() == 1,
            "day {d} should be within 1 tick of probe day {}",
            probe.day(),
        );
        assert!(
            m == probe.month() as i32 || (m - probe.month() as i32).abs() == 1,
            "month {m} should be within 1 tick of probe month {}",
            probe.month(),
        );
    }
}

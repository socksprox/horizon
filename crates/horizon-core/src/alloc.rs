//! `try_reserve` failure marking for the `try_`-suffixed fallible paths.
//!
//! An allocation failure deep inside a generation run must reach the `try_`
//! entry point as `None`, not `handle_alloc_error`. Sites whose growth can
//! hit the wall reserve fallibly, [`note`] the failure, and return the error
//! their signature already carries; the entry point [`clear`]s, runs, and
//! checks [`take`]. A run that failed a reservation always returns its error
//! upward, so a marked run that finished is only noise from a path that
//! recovered anyway — entry points treat `None` as the answer only for an
//! error result.
use std::cell::Cell;

thread_local! {
    static FAILED: Cell<bool> = const { Cell::new(false) };
}

/// Mark that a `try_reserve` failed on this thread. Call right before
/// returning the ordinary error the site already produces.
pub fn note() {
    FAILED.with(|failed| failed.set(true));
}

/// Read-and-clear the mark. `try_` entry points check this after the run.
pub fn take() -> bool {
    FAILED.with(|failed| failed.replace(false))
}

/// Drop a stale mark before a `try_` run.
pub fn clear() {
    FAILED.with(|failed| failed.set(false));
}

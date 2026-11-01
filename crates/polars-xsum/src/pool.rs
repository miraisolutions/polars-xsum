//! The plugin's thread pool.
//!
//! A plugin is a separate shared library with its own copies of rayon and
//! polars-core, so it cannot use the thread pool of the Polars host process.
//! It needs its own pool for parallel work. That pool is:
//!
//! * **dedicated and named** (`xsum-0`, `xsum-1`, …), so its threads are
//!   identifiable in profilers and do not touch rayon's global pool;
//! * **sized like Polars' pool**, from `POLARS_MAX_THREADS` (read with Polars'
//!   own configuration code), defaulting to the number of available CPUs.
//!   `RAYON_NUM_THREADS` overrides it, for this plugin only.
//!
//! When a Polars thread hands work to this pool, it blocks (without spinning)
//! until the work is done. CPU use is therefore bounded by the pool size even
//! when Polars evaluates several `xsum` expressions concurrently.

use std::sync::LazyLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Environment variable that overrides the pool size.
const OVERRIDE_VAR: &str = "RAYON_NUM_THREADS";

/// The plugin's thread pool, created on first use.
static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    let threads = pool_size(
        std::env::var(OVERRIDE_VAR).ok().as_deref(),
        polars_config::config().max_threads(),
    );
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("xsum-{index}"))
        .build()
        .expect("failed to create the xsum thread pool")
});

/// Runs `op` inside the plugin's thread pool.
///
/// # Panics
///
/// Panics if the thread pool cannot be created (on first use).
pub(crate) fn install<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    POOL.install(op)
}

/// Number of threads in the plugin's thread pool (creating it if necessary).
///
/// # Panics
///
/// Panics if the thread pool cannot be created.
pub(crate) fn size() -> usize {
    POOL.current_num_threads()
}

/// Resolves the pool size from the override variable's value (if any) and
/// Polars' configured maximum number of threads.
///
/// A positive integer override wins. Otherwise Polars' setting is used; rayon
/// interprets 0 as "number of available CPUs", just as in Polars.
fn pool_size(override_value: Option<&str>, polars_max_threads: usize) -> usize {
    override_value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(polars_max_threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_polars_by_default() {
        assert_eq!(pool_size(None, 12), 12);
    }

    #[test]
    fn positive_override_wins() {
        assert_eq!(pool_size(Some("3"), 12), 3);
        assert_eq!(pool_size(Some(" 7 "), 12), 7);
    }

    #[test]
    fn invalid_or_zero_override_is_ignored() {
        assert_eq!(pool_size(Some("0"), 12), 12);
        assert_eq!(pool_size(Some("many"), 12), 12);
        assert_eq!(pool_size(Some("-2"), 12), 12);
        assert_eq!(pool_size(Some(""), 12), 12);
    }

    #[test]
    fn pool_threads_are_named() {
        let name = install(|| std::thread::current().name().map(str::to_owned));
        assert!(name.is_some_and(|n| n.starts_with("xsum-")));
        assert!(size() >= 1);
    }
}

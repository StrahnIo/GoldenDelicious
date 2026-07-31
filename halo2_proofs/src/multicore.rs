//! An interface for dealing with the kinds of parallel computations involved in
//! `halo2`. It's currently just a (very!) thin wrapper around `rayon` but may
//! be extended in the future to allow for various parallelism strategies.

#![allow(unsafe_code)] // thread QoS / affinity requires libc syscalls here

#[cfg(all(
    feature = "multicore",
    target_arch = "wasm32",
    not(target_feature = "atomics")
))]
compile_error!(
    "The multicore feature flag is not supported on wasm32 architectures without atomics"
);

pub use maybe_rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    join, scope,
};

#[cfg(feature = "multicore")]
pub use maybe_rayon::{current_num_threads, iter::IndexedParallelIterator};

use std::sync::OnceLock;

/// Number of performance (P) cores on this machine.
///
/// On Apple Silicon, `hw.perflevel0.logicalcpu` reports the P-core count;
/// elsewhere, fall back to all available cores.
pub fn num_p_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        let mut val: u32 = 0;
        let mut len = std::mem::size_of::<u32>();
        let ok = unsafe {
            libc::sysctlbyname(
                b"hw.perflevel0.logicalcpu\0".as_ptr() as *const libc::c_char,
                &mut val as *mut _ as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } == 0;
        if ok && val > 0 {
            return val as usize;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Run `f` on a rayon pool restricted to the performance cores.
///
/// The recursive evaluator dispatches its chunk work here so the slower
/// efficiency cores don't drag down the wall time. Workers request
/// `USER_INTERACTIVE` QoS (which the Apple Silicon scheduler biases toward the
/// P-cores). Falls back to a plain call when `multicore` is disabled.
pub fn install_on_p_cores<R>(f: impl FnOnce() -> R + Send) -> R
where
    R: Send,
{
    #[cfg(feature = "multicore")]
    {
        static POOL: OnceLock<maybe_rayon::ThreadPool> = OnceLock::new();
        let pool = POOL.get_or_init(|| {
            let n = num_p_cores();
            maybe_rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .spawn_handler(|thread| {
                    std::thread::Builder::new()
                        .name(thread.name().unwrap_or("rayon-worker").to_string())
                        .spawn(move || {
                            #[cfg(target_os = "macos")]
                            unsafe {
                                // Biases the scheduler toward P-cores.
                                libc::pthread_set_qos_class_self_np(
                                    libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE,
                                    0,
                                );
                            }
                            thread.run();
                        })
                        .map(|_| ())
                })
                .build()
                .expect("failed to build performance-core rayon pool")
        });
        pool.install(f)
    }
    #[cfg(not(feature = "multicore"))]
    {
        f()
    }
}

#[cfg(not(feature = "multicore"))]
pub fn current_num_threads() -> usize {
    1
}

#[cfg(not(feature = "multicore"))]
pub trait IndexedParallelIterator: std::iter::Iterator {}

pub trait TryFoldAndReduce<T, E> {
    /// Implements `iter.try_fold().try_reduce()` for `rayon::iter::ParallelIterator`,
    /// falling back on `Iterator::try_fold` when the `multicore` feature flag is
    /// disabled.
    /// The `try_fold_and_reduce` function can only be called by a iter with
    /// `Result<T, E>` item type because the `fold_op` must meet the trait
    /// bounds of both `try_fold` and `try_reduce` from rayon.   
    fn try_fold_and_reduce(
        self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E>;
}

#[cfg(feature = "multicore")]
impl<T, E, I> TryFoldAndReduce<T, E> for I
where
    T: Send + Sync,
    E: Send + Sync,
    I: maybe_rayon::iter::ParallelIterator<Item = Result<T, E>>,
{
    fn try_fold_and_reduce(
        self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E> {
        self.try_fold(&identity, &fold_op)
            .try_reduce(&identity, |a, b| fold_op(a, Ok(b)))
    }
}

#[cfg(not(feature = "multicore"))]
impl<T, E, I> TryFoldAndReduce<T, E> for I
where
    I: std::iter::Iterator<Item = Result<T, E>>,
{
    fn try_fold_and_reduce(
        mut self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E> {
        self.try_fold(identity(), fold_op)
    }
}

pub(crate) trait TheBestReduce {
    type Item;

    /// Combines the best of `std::iter` and `rayon` reductions.
    fn the_best_reduce(
        self,
        identity: impl Fn() -> Self::Item + Send + Sync,
        op: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item>;
}

#[cfg(feature = "multicore")]
impl<I> TheBestReduce for I
where
    I: maybe_rayon::iter::ParallelIterator,
{
    type Item = <Self as maybe_rayon::iter::ParallelIterator>::Item;

    fn the_best_reduce(
        self,
        identity: impl Fn() -> Self::Item + Send + Sync,
        op: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item> {
        Some(self.reduce(identity, op))
    }
}

#[cfg(not(feature = "multicore"))]
impl<I> TheBestReduce for I
where
    I: std::iter::Iterator,
{
    type Item = <Self as std::iter::Iterator>::Item;

    fn the_best_reduce(
        self,
        _: impl Fn() -> Self::Item + Send + Sync,
        f: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item> {
        self.reduce(f)
    }
}

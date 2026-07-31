/// Errors returned by Fuji operations.
///
/// Maps directly from the C error codes in `libfuji.dylib`:
///
/// | Code | Variant | Cause |
/// |------|---------|-------|
/// | `-2` | `AmxUnavailable` | AMX not present (Intel, VM, or old hardware) |
/// | `-3` | `OutOfMemory` | `calloc` or `malloc` returned NULL |
/// | `-4` | `InvalidContext` | NULL pointer or uninitialised streaming MSM state |
/// | other | `Generic` | Unknown or unexpected error |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FujiError {
    /// AMX coprocessor is not available. Fallback scalar code was used.
    AmxUnavailable,

    /// A required pointer argument was NULL, or a streaming MSM state was not
    /// initialised before calling `feed` or `finalize`.
    InvalidContext,

    /// PRL precondition violated (e.g. identity accumulator passed to
    /// `prl_add_mixed_3`, or colliding/zero window indices).
    InvalidInput,

    /// Dynamic memory allocation failed.
    OutOfMemory,

    /// An unspecified error occurred.
    Generic,
}

impl From<i32> for FujiError {
    fn from(code: i32) -> Self {
        match code {
            -2 => FujiError::AmxUnavailable,
            -3 => FujiError::OutOfMemory,
            -4 => FujiError::InvalidContext,
            _ => FujiError::Generic,
        }
    }
}

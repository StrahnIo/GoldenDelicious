//! SME (Scalable Matrix Extension) primitives.
//!
//! Provides low-level access to Apple M4's SME coprocessor for matrix
//! outer-product operations. The main entry point is [`umopa_outer_product`]
//! which computes a 32×32 8-bit unsigned outer product in a single instruction.
//!
//! # Streaming mode
//!
//! SME requires entering *streaming SVE mode* via [`SmeStream::enter`] before
//! ZA tile or SME instructions can be used. The returned guard automatically
//! exits streaming mode when dropped.
//!
//! # Toolchain limitation
//!
//! ZA tile extraction (reading results back from the tile to memory) uses
//! `.word`-encoded STR ZA instructions. These encodings were derived from
//! Apple M4 hardware probing and may not work on all toolchains. If the
//! encoding is incorrect, the process will receive `SIGILL`.

use core::arch::asm;

/// Enters streaming SVE/SME mode by executing `smstart`.
///
/// # Safety
/// Must not already be in streaming mode (nesting is not supported).
pub unsafe fn enter_streaming() {
    asm!(".arch armv9-a+sme", "smstart", options(nostack, preserves_flags));
}

/// Exits streaming SVE/SME mode by executing `smstop`.
///
/// # Safety
/// Must be called only after a matching `enter_streaming()`.
pub unsafe fn exit_streaming() {
    asm!(".arch armv9-a+sme", "smstop", options(nostack, preserves_flags));
}

/// A RAII guard that holds the CPU in SME streaming mode.
///
/// When dropped, automatically exits streaming mode via `smstop`.
/// Prevents mismatched enter/exit pairs.
///
/// # Example
///
/// ```ignore
/// let _stream = SmeStream::enter();
/// // ZA tile and SME instructions are now accessible
/// // ... do SME work ...
/// // drop(_stream) → smstop is called automatically
/// ```
#[must_use = "streaming mode exits immediately if the guard is not held"]
pub struct SmeStream(());

impl SmeStream {
    /// Enter streaming SVE/SME mode.
    ///
    /// Returns a guard that exits on drop. Panics if SME is not available
    /// on this CPU. Panics if already in streaming mode.
    pub fn enter() -> Self {
        if !super::detection::sme_available() {
            panic!("SME is not available on this CPU");
        }
        unsafe { enter_streaming() }
        SmeStream(())
    }

    /// Returns `true` if the current thread holds a streaming-mode guard alive.
    /// Note: this is a thread-local flag, not the hardware PSTATE.
    pub fn is_active() -> bool {
        STREAMING_DEPTH.with(|d| d.get() > 0)
    }
}

impl Drop for SmeStream {
    fn drop(&mut self) {
        unsafe { exit_streaming() }
    }
}

std::thread_local! {
    static STREAMING_DEPTH: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}

/// Compute a 32×32 8-bit unsigned outer product using SME UMOPA.
///
/// Returns `1024` elements (as `[u32; 1024]`) where `result[j * 32 + i]`
/// represents the product `a[i] * b[j]` accumulated over the 8-bit inputs.
///
/// This function enters streaming mode internally, executes UMOPA, extracts
/// the ZA tile to memory, and exits streaming mode.
///
/// # Panics
///
/// - Panics if SME is not available on this CPU.
/// - Panics if the ZA tile extraction encoding is not supported on this
///   hardware/toolchain (the process may also receive `SIGILL`).
pub fn umopa_outer_product(a: &[u8; 32], b: &[u8; 32]) -> [u32; 1024] {
    let mut out = [0u32; 1024];
    unsafe {
        umopa_outer_product_raw(a, b, &mut out);
    }
    out
}

/// Raw variant of [`umopa_outer_product`].
///
/// # Safety
///
/// - `a` and `b` must be 32-byte slices.
/// - `out` must be at least 1024 elements (4096 bytes) and 64-byte aligned.
/// - Must NOT be in streaming mode (this function enters and exits internally).
pub unsafe fn umopa_outer_product_raw(a: &[u8; 32], b: &[u8; 32], out: &mut [u32; 1024]) {
    assert!(out.len() >= 1024, "output buffer too small");
    let out_ptr = out.as_mut_ptr();

    asm!(
        ".arch armv9-a+sme",
        "smstart",

        "ptrue  p0.b",
        "mov    x2, #0",
        "mov    x3, #32",
        "whilelt p1.b, x2, x3",
        "ld1b   {{z0.b}}, p1/z, [{a}]",
        "ld1b   {{z1.b}}, p1/z, [{b}]",
        "umopa  za0.s, p0/m, p0/m, z0.b, z1.b",
        "mov    x22, {out}",

        ".word  0xe1200016", ".word  0xe1202016",
        ".word  0xe1204016", ".word  0xe1206016",
        ".word  0xe1208016", ".word  0xe120a016",
        ".word  0xe120c016", ".word  0xe120e016",
        ".word  0xe1200416", ".word  0xe1202416",
        ".word  0xe1204416", ".word  0xe1206416",
        ".word  0xe1208416", ".word  0xe120a416",
        ".word  0xe120c416", ".word  0xe120e416",
        ".word  0xe1200816", ".word  0xe1202816",
        ".word  0xe1204816", ".word  0xe1206816",
        ".word  0xe1208816", ".word  0xe120a816",
        ".word  0xe120c816", ".word  0xe120e816",
        ".word  0xe1200c16", ".word  0xe1202c16",
        ".word  0xe1204c16", ".word  0xe1206c16",
        ".word  0xe1208c16", ".word  0xe120ac16",
        ".word  0xe120cc16", ".word  0xe120ec16",
        ".word  0xe1201016", ".word  0xe1203016",
        ".word  0xe1205016", ".word  0xe1207016",
        ".word  0xe1209016", ".word  0xe120b016",
        ".word  0xe120d016", ".word  0xe120f016",
        ".word  0xe1201416", ".word  0xe1203416",
        ".word  0xe1205416", ".word  0xe1207416",
        ".word  0xe1209416", ".word  0xe120b416",
        ".word  0xe120d416", ".word  0xe120f416",
        ".word  0xe1201816", ".word  0xe1203816",
        ".word  0xe1205816", ".word  0xe1207816",
        ".word  0xe1209816", ".word  0xe120b816",
        ".word  0xe120d816", ".word  0xe120f816",
        ".word  0xe1201c16", ".word  0xe1203c16",
        ".word  0xe1205c16", ".word  0xe1207c16",
        ".word  0xe1209c16", ".word  0xe120bc16",
        ".word  0xe120dc16", ".word  0xe120fc16",

        "smstop",

        a = in(reg) a.as_ptr(),
        b = in(reg) b.as_ptr(),
        out = in(reg) out_ptr,
        out("x2") _, out("x3") _,
        out("p0") _, out("p1") _,
        out("z0") _, out("z1") _,
        options(nostack, preserves_flags),
    );
}

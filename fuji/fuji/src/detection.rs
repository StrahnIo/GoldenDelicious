use std::ffi::CStr;

/// Returns `true` if the AMX matrix coprocessor is available and usable.
///
/// On Apple Silicon (M1–M4) this returns `true`. On Intel Macs, VMs,
/// or older hardware it returns `false`.
pub fn amx_available() -> bool {
    unsafe { fuji_sys::fuji_detect_amx() == 0 }
}

/// Returns `true` if the SME (Scalable Matrix Extension) is available.
///
/// SME is available on Apple M4 and later. It provides an alternative
/// path for matrix outer products via the streaming SVE ZA tile.
pub fn sme_available() -> bool {
    unsafe { fuji_sys::fuji_sme_available() != 0 }
}

/// Returns the CPU brand string (e.g. `"Apple M4"`, `"Apple M3 Pro"`).
///
/// The returned string has `'static` lifetime and is never empty or null.
pub fn cpu_brand() -> &'static str {
    unsafe {
        let ptr = fuji_sys::fuji_cpu_brand();
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

/// Returns the library version as a semver string (`"0.1.0"`).
///
/// The returned string has `'static` lifetime.
pub fn lib_version() -> &'static str {
    unsafe {
        let ptr = fuji_sys::fuji_lib_version();
        CStr::from_ptr(ptr).to_str().unwrap_or("0.0.0")
    }
}


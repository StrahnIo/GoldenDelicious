/// Returns `true` if the PRL parallel execution extension is available.
pub fn prl_available() -> bool {
    unsafe { fuji_sys::fuji_prl_available() != 0 }
}

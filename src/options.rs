use std::error::Error;
use std::ffi::{c_void, CStr, CString};
use std::fmt::{Debug, Display};
use std::os::raw::{c_char, c_int};

/// An error occurred while trying to set a model option
#[derive(Debug, Clone)]
pub struct TrySetOptionError;

impl Display for TrySetOptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error setting model option")
    }
}

impl Error for TrySetOptionError {}

/// A trait defining the possible value types for HiGHS model options
pub trait HighsOptionValue {
    /// Apply the given value to the given option for the HiGHS model
    ///
    /// # Safety
    ///
    /// This function should only be called with valid pointers to `highs` and `option`. `option`
    /// should be a NUL-terminated C-string.
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int;
}

impl HighsOptionValue for bool {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setBoolOptionValue(highs, option, if self { 1 } else { 0 })
    }
}

impl HighsOptionValue for i32 {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setIntOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for f64 {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setDoubleOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for &CStr {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setStringOptionValue(highs, option, self.as_ptr())
    }
}

impl HighsOptionValue for &[u8] {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        CString::new(self)
            .expect("invalid highs option value")
            .apply_to_highs(highs, option)
    }
}

impl HighsOptionValue for &str {
    unsafe fn apply_to_highs(self, highs: *mut c_void, option: *const c_char) -> c_int {
        self.as_bytes().apply_to_highs(highs, option)
    }
}

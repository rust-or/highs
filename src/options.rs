use std::ffi::{c_void, CString, CStr};
use std::os::raw::{c_char, c_int};

pub trait HighsOptionValue {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int;
}

impl HighsOptionValue for bool {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setBoolOptionValue(highs, option, if self { 1 } else { 0 })
    }
}

impl HighsOptionValue for i32 {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setIntOptionValue(highs, option, self)
    }
}

impl HighsOptionValue for f64 {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setDoubleOptionValue(highs, option, self)
    }
}

impl<'a> HighsOptionValue for &'a CStr {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        highs_sys::Highs_setStringOptionValue(highs, option, self.as_ptr())
    }
}

impl<'a> HighsOptionValue for &'a [u8] {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        CString::new(self).expect("invalid highs option value").set_option(highs, option)
    }
}

impl<'a> HighsOptionValue for &'a str {
    unsafe fn set_option(self, highs: *mut c_void, option: *const c_char) -> c_int {
        self.as_bytes().set_option(highs, option)
    }
}

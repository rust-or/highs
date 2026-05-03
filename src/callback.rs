//! # User Callbacks in HiGHS

use std::ffi::{c_int, c_void};

/// User callbacks while solving
pub trait Callback {
    /// The main callback routine
    fn callback(&mut self, context: CallbackOuterContext<'_>) -> CallbackReturn;
}

/// The context of a user callback
pub struct CallbackOuterContext<'a> {
    data: &'a highs_sys::HighsCallbackDataOut,
    callback_type: c_int,
}

// Applicable in all contexts
impl<'a> CallbackOuterContext<'a> {
    /// Gets the inner callback context
    pub fn inner(self) -> CallbackType<'a> {
        match self.callback_type {
            highs_sys::kHighsCallbackLogging => CallbackType::Logging(CallbackContext {
                data: self.data,
                _ctx: CbCtxLogging,
            }),
            highs_sys::kHighsCallbackSimplexInterrupt => {
                CallbackType::SimplexInterrupt(CallbackContext {
                    data: self.data,
                    _ctx: CbCtxSimplexInterrupt,
                })
            }
            highs_sys::kHighsCallbackIpmInterrupt => CallbackType::IpmInterrupt(CallbackContext {
                data: self.data,
                _ctx: CbCtxIpmInterrupt,
            }),
            highs_sys::kHighsCallbackMipSolution => CallbackType::MipSolution(CallbackContext {
                data: self.data,
                _ctx: CbCtxMipSolution,
            }),
            highs_sys::kHighsCallbackMipImprovingSolution => {
                CallbackType::MipImprovingSolution(CallbackContext {
                    data: self.data,
                    _ctx: CbCtxMipImprovingSolution,
                })
            }
            highs_sys::kHighsCallbackMipLogging => CallbackType::MipLogging(CallbackContext {
                data: self.data,
                _ctx: CbCtxMipLogging,
            }),
            highs_sys::kHighsCallbackMipInterrupt => CallbackType::MipInterrupt(CallbackContext {
                data: self.data,
                _ctx: CbCtxMipInterrupt,
            }),
            highs_sys::kHighsCallbackMipGetCutPool => {
                CallbackType::MipGetCutPool(CallbackContext {
                    data: self.data,
                    _ctx: CbCtxMipGetCutPool,
                })
            }
            highs_sys::kHighsCallbackMipDefineLazyConstraints => {
                CallbackType::MipDefineLazyConstraints(CallbackContext {
                    data: self.data,
                    _ctx: CbCtxMipDefineLazyConstraints,
                })
            }
            _ => unreachable!(),
        }
    }

    /// Gets the running time of the solver
    pub fn get_running_time(&self) -> f64 {
        self.data.running_time
    }
}

/// The type of callback
pub enum CallbackType<'a> {
    /// Logging callback
    Logging(CallbackContext<'a, CbCtxLogging>),
    /// Simplex interrupt callback
    SimplexInterrupt(CallbackContext<'a, CbCtxSimplexInterrupt>),
    /// IPM interrupt callback
    IpmInterrupt(CallbackContext<'a, CbCtxIpmInterrupt>),
    /// Found a MIP solution
    MipSolution(CallbackContext<'a, CbCtxMipSolution>),
    /// Found an improving MIP solution
    MipImprovingSolution(CallbackContext<'a, CbCtxMipImprovingSolution>),
    /// MIP logging callback
    MipLogging(CallbackContext<'a, CbCtxMipLogging>),
    /// MIP interrupt callback
    MipInterrupt(CallbackContext<'a, CbCtxMipInterrupt>),
    /// MIP get cut pool callback
    MipGetCutPool(CallbackContext<'a, CbCtxMipGetCutPool>),
    /// MIP define lazy constraints callback
    MipDefineLazyConstraints(CallbackContext<'a, CbCtxMipDefineLazyConstraints>),
}

/// Logging callback context
pub struct CbCtxLogging;
/// Simplex interrupt callback context
pub struct CbCtxSimplexInterrupt;
/// IPM interrupt callback context
pub struct CbCtxIpmInterrupt;
/// MIP solution callback context
pub struct CbCtxMipSolution;
/// MIP improving solution callback context
pub struct CbCtxMipImprovingSolution;
/// MIP logging callback context
pub struct CbCtxMipLogging;
/// MIP interrupt callback context
pub struct CbCtxMipInterrupt;
/// MIP get cut pool callback context
pub struct CbCtxMipGetCutPool;
/// MIP define lazy constraints callback context
pub struct CbCtxMipDefineLazyConstraints;

/// An inner callback context
pub struct CallbackContext<'a, Ctx> {
    data: &'a highs_sys::HighsCallbackDataOut,
    _ctx: Ctx,
}

// Applicable in all contexts
impl<Ctx> CallbackContext<'_, Ctx> {
    /// Gets the running time of the solver
    pub fn get_running_time(&self) -> f64 {
        self.data.running_time
    }
}

/// The return type for a user callback
#[derive(Debug, Default)]
pub struct CallbackReturn {
    user_interrupt: bool,
}

impl CallbackReturn {
    /// Sets the user interrupt value
    pub fn set_interrupt(&mut self, interrupt: bool) -> &mut Self {
        self.user_interrupt = interrupt;
        self
    }
}

pub(crate) struct UserCallbackData<'a>(pub &'a mut dyn Callback);

pub(crate) unsafe extern "C" fn callback(
    callback_type: c_int,
    _message: *const i8,
    out_data: *const highs_sys::HighsCallbackDataOut,
    in_data: *mut highs_sys::HighsCallbackDataIn,
    user_callback_data: *mut c_void,
) {
    let user_callback_data = &mut *user_callback_data.cast::<UserCallbackData>();
    let ctx = CallbackOuterContext {
        data: &*out_data,
        callback_type,
    };
    let res = user_callback_data.0.callback(ctx);
    if res.user_interrupt {
        (*in_data).user_interrupt = 1;
    }
}

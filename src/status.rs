use std::convert::TryFrom;
use std::fmt::{Debug, Formatter};
use std::os::raw::c_int;

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsModelStatus {
    NotSet = 0,
    LoadError = 1,
    ModelError = 2,
    PresolveError = 3,
    SolveError = 4,
    PostsolveError = 5,
    ModelEmpty = 6,
    PrimalInfeasible = 7,
    PrimalUnbounded = 8,
    Optimal = 9,
    ReachedDualObjectiveValueUpperBound = 10,
    ReachedTimeLimit = 11,
    ReachedIterationLimit = 12,
    PrimalDualInfeasible = 13,
    DualInfeasible = 14,
}

#[derive(PartialEq, Clone, Copy)]
pub struct InvalidStatus(pub c_int);

impl Debug for InvalidStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} is not a valid HiGHS model status. \
        This error comes from a bug in highs rust bindings. \
        Please report it.",
            self.0
        )
    }
}

impl TryFrom<c_int> for HighsModelStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        use highs_sys::*;
        match value {
            MODEL_STATUS_NOTSET => Ok(Self::NotSet),
            MODEL_STATUS_LOAD_ERROR => Ok(Self::LoadError),
            MODEL_STATUS_MODEL_ERROR => Ok(Self::ModelError),
            MODEL_STATUS_PRESOLVE_ERROR => Ok(Self::PresolveError),
            MODEL_STATUS_SOLVE_ERROR => Ok(Self::SolveError),
            MODEL_STATUS_POSTSOLVE_ERROR => Ok(Self::PostsolveError),
            MODEL_STATUS_MODEL_EMPTY => Ok(Self::ModelEmpty),
            MODEL_STATUS_PRIMAL_INFEASIBLE => Ok(Self::PrimalInfeasible),
            MODEL_STATUS_PRIMAL_UNBOUNDED => Ok(Self::PrimalUnbounded),
            MODEL_STATUS_OPTIMAL => Ok(Self::Optimal),
            MODEL_STATUS_REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND => {
                Ok(Self::ReachedDualObjectiveValueUpperBound)
            }
            MODEL_STATUS_REACHED_TIME_LIMIT => Ok(Self::ReachedTimeLimit),
            MODEL_STATUS_REACHED_ITERATION_LIMIT => Ok(Self::ReachedIterationLimit),
            MODEL_STATUS_PRIMAL_DUAL_INFEASIBLE => Ok(Self::PrimalDualInfeasible),
            MODEL_STATUS_DUAL_INFEASIBLE => Ok(Self::DualInfeasible),
            n => Err(InvalidStatus(n)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsStatus {
    OK = 0,
    Warning = 1,
    Error = 2,
}

impl TryFrom<c_int> for HighsStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, InvalidStatus> {
        use highs_sys::*;
        match value {
            STATUS_OK => Ok(Self::OK),
            STATUS_WARNING => Ok(Self::Warning),
            STATUS_ERROR => Ok(Self::Error),
            n => Err(InvalidStatus(n)),
        }
    }
}

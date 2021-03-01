use std::convert::TryFrom;
use std::fmt::{Debug, Formatter};
use std::os::raw::c_int;

/// The kinds of results of an optimization
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsModelStatus {
    /// not initialized
    NotSet = 0,
    /// Unable to load model
    LoadError = 1,
    /// invalid model
    ModelError = 2,
    /// Unable to run the pre-solve phase
    PresolveError = 3,
    /// Unable to solve
    SolveError = 4,
    /// Unable to clean after solve
    PostsolveError = 5,
    /// No variables in the model: nothing to optimize
    /// ```
    /// use highs::*;
    /// let solved = ColProblem::new().optimise(Sense::Maximise).solve();
    /// assert_eq!(solved.status(), HighsModelStatus::ModelEmpty);
    /// ```
    ModelEmpty = 6,
    /// There is no solution to the problem
    PrimalInfeasible = 7,
    /// The problem is unbounded: there is no single optimal value
    PrimalUnbounded = 8,
    /// An optimal solution was found
    Optimal = 9,
    /// reached limit
    ReachedDualObjectiveValueUpperBound = 10,
    /// reached limit
    ReachedTimeLimit = 11,
    /// reached limit
    ReachedIterationLimit = 12,
    /// cannot solve dual
    PrimalDualInfeasible = 13,
    /// cannot solve dual
    DualInfeasible = 14,
}

/// This error should never happen: an unexpected status was returned
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

/// The status of a highs operation
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsStatus {
    /// Success
    OK = 0,
    /// Done, with warning
    Warning = 1,
    /// An error occurred
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

use std::convert::TryFrom;
use std::fmt::{Debug, Formatter};
use std::num::TryFromIntError;
use std::os::raw::c_int;

use highs_sys::*;

/// The kinds of results of an optimization
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
#[non_exhaustive]
pub enum HighsModelStatus {
    /// not initialized
    NotSet = MODEL_STATUS_NOTSET as isize,
    /// Unable to load model
    LoadError = MODEL_STATUS_LOAD_ERROR as isize,
    /// invalid model
    ModelError = MODEL_STATUS_MODEL_ERROR as isize,
    /// Unable to run the pre-solve phase
    PresolveError = MODEL_STATUS_PRESOLVE_ERROR as isize,
    /// Unable to solve
    SolveError = MODEL_STATUS_SOLVE_ERROR as isize,
    /// Unable to clean after solve
    PostsolveError = MODEL_STATUS_POSTSOLVE_ERROR as isize,
    /// No variables in the model: nothing to optimize
    /// ```
    /// use highs::*;
    /// let solved = ColProblem::new().optimise(Sense::Maximise).solve();
    /// assert_eq!(solved.status(), HighsModelStatus::ModelEmpty);
    /// ```
    ModelEmpty = MODEL_STATUS_MODEL_EMPTY as isize,
    /// There is no solution to the problem
    Infeasible = MODEL_STATUS_INFEASIBLE as isize,
    /// The problem in unbounded or infeasible
    UnboundedOrInfeasible = MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE as isize,
    /// The problem is unbounded: there is no single optimal value
    Unbounded = MODEL_STATUS_UNBOUNDED as isize,
    /// An optimal solution was found
    Optimal = MODEL_STATUS_OPTIMAL as isize,
    /// objective bound
    ObjectiveBound = MODEL_STATUS_OBJECTIVE_BOUND as isize,
    /// objective target
    ObjectiveTarget = MODEL_STATUS_OBJECTIVE_TARGET as isize,
    /// reached time limit
    ReachedTimeLimit = MODEL_STATUS_REACHED_TIME_LIMIT as isize,
    /// reached iteration limit
    ReachedIterationLimit = MODEL_STATUS_REACHED_ITERATION_LIMIT as isize,
    /// Unknown model status
    Unknown = MODEL_STATUS_UNKNOWN as isize,
    /// reached solution limit
    ReachedSolutionLimit = MODEL_STATUS_REACHED_SOLUTION_LIMIT as isize,
    /// interrupted
    ReachedInterrupt = MODEL_STATUS_REACHED_INTERRUPT as isize,
    /// reached memory limit
    ReachedMemoryLimit = MODEL_STATUS_REACHED_MEMORY_LIMIT as isize,
    /// Interrupted internally by HiGHS (distinct from a user interrupt).
    HighsInterrupt = MODEL_STATUS_HIGHS_INTERRUPT as isize,
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

impl HighsModelStatus {
    /// Interpret status codes from newer HiGHS releases without panicking.
    ///
    /// Keep TryFrom strict so callers can still detect unsupported codes.
    pub(crate) fn from_raw_or_unknown(value: c_int) -> Self {
        match Self::try_from(value) {
            Ok(status) => status,
            Err(_) => {
                log::warn!("Unrecognized HiGHS model status {value}; treating as Unknown");
                Self::Unknown
            }
        }
    }
}

impl TryFrom<c_int> for HighsModelStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            MODEL_STATUS_NOTSET => Ok(Self::NotSet),
            MODEL_STATUS_LOAD_ERROR => Ok(Self::LoadError),
            MODEL_STATUS_MODEL_ERROR => Ok(Self::ModelError),
            MODEL_STATUS_PRESOLVE_ERROR => Ok(Self::PresolveError),
            MODEL_STATUS_SOLVE_ERROR => Ok(Self::SolveError),
            MODEL_STATUS_POSTSOLVE_ERROR => Ok(Self::PostsolveError),
            MODEL_STATUS_MODEL_EMPTY => Ok(Self::ModelEmpty),
            MODEL_STATUS_INFEASIBLE => Ok(Self::Infeasible),
            MODEL_STATUS_UNBOUNDED => Ok(Self::Unbounded),
            MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => Ok(Self::UnboundedOrInfeasible),
            MODEL_STATUS_OPTIMAL => Ok(Self::Optimal),
            MODEL_STATUS_OBJECTIVE_BOUND => Ok(Self::ObjectiveBound),
            MODEL_STATUS_OBJECTIVE_TARGET => Ok(Self::ObjectiveTarget),
            MODEL_STATUS_REACHED_TIME_LIMIT => Ok(Self::ReachedTimeLimit),
            MODEL_STATUS_REACHED_ITERATION_LIMIT => Ok(Self::ReachedIterationLimit),
            MODEL_STATUS_UNKNOWN => Ok(Self::Unknown),
            MODEL_STATUS_REACHED_SOLUTION_LIMIT => Ok(Self::ReachedSolutionLimit),
            MODEL_STATUS_REACHED_INTERRUPT => Ok(Self::ReachedInterrupt),
            MODEL_STATUS_REACHED_MEMORY_LIMIT => Ok(Self::ReachedMemoryLimit),
            MODEL_STATUS_HIGHS_INTERRUPT => Ok(Self::HighsInterrupt),
            n => Err(InvalidStatus(n))
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

impl From<TryFromIntError> for HighsStatus {
    fn from(_: TryFromIntError) -> Self {
        Self::Error
    }
}

impl TryFrom<c_int> for HighsStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, InvalidStatus> {
        match value {
            STATUS_OK => Ok(Self::OK),
            STATUS_WARNING => Ok(Self::Warning),
            STATUS_ERROR => Ok(Self::Error),
            n => Err(InvalidStatus(n)),
        }
    }
}

/// The status of a solution
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Ord, Eq)]
pub enum HighsSolutionStatus {
    /// No solution found
    None = SOLUTION_STATUS_NONE as isize,
    /// No solution exists
    Infeasible = SOLUTION_STATUS_INFEASIBLE as isize,
    /// A feasible solution was found
    Feasible = SOLUTION_STATUS_FEASIBLE as isize,
}

impl TryFrom<c_int> for HighsSolutionStatus {
    type Error = InvalidStatus;

    fn try_from(value: c_int) -> Result<Self, Self::Error> {
        match value {
            SOLUTION_STATUS_NONE => Ok(Self::None),
            SOLUTION_STATUS_INFEASIBLE => Ok(Self::Infeasible),
            SOLUTION_STATUS_FEASIBLE => Ok(Self::Feasible),
            n => Err(InvalidStatus(n)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HighsModelStatus;
    use highs_sys::{MODEL_STATUS_MAX, MODEL_STATUS_MIN};

    #[test]
    fn decodes_all_model_statuses_from_highs_sys() {
        // HiGHS 1.16 added status 19 (kHighsInterrupt). Before the fix,
        // decoding it returned InvalidStatus and SolvedModel::status() panicked.
        for code in MODEL_STATUS_MIN..=MODEL_STATUS_MAX {
            assert!(
                HighsModelStatus::try_from(code).is_ok(),
                "unrecognized HiGHS model status {code}"
            );
        }
    }

    #[test]
    fn unknown_future_model_status_does_not_panic() {
        use highs_sys::MODEL_STATUS_MAX;

        let future_code = MODEL_STATUS_MAX + 1;
        assert!(HighsModelStatus::try_from(future_code).is_err());
        assert_eq!(
            HighsModelStatus::from_raw_or_unknown(future_code),
            HighsModelStatus::Unknown
        );
    }
}

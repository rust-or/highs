#![forbid(missing_docs)]
//! Safe rust binding to the [HiGHS](https://highs.dev) linear programming solver.
//!
//! ## Usage example
//!
//! ### Building a problem constraint by constraint with [RowProblem]
//!
//! Useful for traditional problem modelling where you first declare your variables, then add
//!constraints one by one.
//!
//! ```
//! use highs::{Sense, Model, HighsModelStatus, RowProblem};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = RowProblem::default();
//! // Create a variable named x, with a coefficient of 1 in the objective function,
//! // that is bound between 0 and +∞.
//! let x = pb.add_column(1., 0..);
//! let y = pb.add_column(2., 0..);
//! let z = pb.add_column(1., 0..);
//! // constraint c1: x*3 + y*1 is bound to ]-∞; 6]
//! pb.add_row(..=6, &[(x, 3.), (y, 1.)]);
//! // constraint c2: y*1 +  z*2 is bound to ]-∞; 7]
//! pb.add_row(..=7, &[(y, 1.), (z, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
//!
//! ### Building a problem variable by variable with [ColProblem]
//!
//! Useful for resource allocation problems and other problems when you know in advance the number
//! of constraints and their bounds, but dynamically add new variables to the problem.
//!
//! This is slightly more efficient than building the problem constraint by constraint.
//!
//! ```
//! use highs::{ColProblem, Sense};
//! let mut pb = ColProblem::new();
//! // We cannot use more then 5 units of sugar in total.
//! let sugar = pb.add_row(..=5);
//! // We cannot use more then 3 units of milk in total.
//! let milk = pb.add_row(..=3);
//! // We have a first cake that we can sell for 2€. Baking it requires 1 unit of milk and 2 of sugar.
//! pb.add_column(2., 0.., &[(sugar, 2.), (milk, 1.)]);
//! // We have a second cake that we can sell for 8€. Baking it requires 2 units of milk and 3 of sugar.
//! pb.add_column(8., 0.., &[(sugar, 3.), (milk, 2.)]);
//! // Find the maximal possible profit
//! let solution = pb.optimise(Sense::Maximise).solve().get_solution();
//! // The solution is to bake only 1.5 portions of the second cake
//! assert_eq!(solution.columns(), vec![0.,1.5]);
//! ```
//!
//! ```
//! use highs::{Sense, Model, HighsModelStatus, ColProblem};
//! // max: x + 2y + z
//! // under constraints:
//! // c1: 3x +  y      <= 6
//! // c2:       y + 2z <= 7
//! let mut pb = ColProblem::default();
//! let c1 = pb.add_row(..6.);
//! let c2 = pb.add_row(..7.);
//! // x
//! pb.add_column(1., 0.., &[(c1, 3.)]);
//! // y
//! pb.add_column(2., 0.., &[(c1, 1.), (c2, 1.)]);
//! // z
//! pb.add_column(1., 0.., vec![(c2, 2.)]);
//!
//! let solved = pb.optimise(Sense::Maximise).solve();
//!
//! assert_eq!(solved.status(), HighsModelStatus::Optimal);
//!
//! let solution = solved.get_solution();
//! // The expected solution is x=0  y=6  z=0.5
//! assert_eq!(solution.columns(), vec![0., 6., 0.5]);
//! // All the constraints are at their maximum
//! assert_eq!(solution.rows(), vec![6., 7.]);
//! ```
use std::convert::{TryFrom, TryInto};
use std::ffi::{c_void, CString};
use std::ops::{Bound, RangeBounds};
use std::os::raw::c_int;

use highs_sys::*;

use crate::options::HighsOptionValue;
pub use matrix_col::{ColMatrix, Row};
pub use matrix_row::{Col, RowMatrix};
pub use status::{HighsModelStatus, HighsStatus};

/// A problem where variables are declared first, and constraints are then added dynamically.
/// See [`Problem<RowMatrix>`](Problem#impl-1).
pub type RowProblem = Problem<RowMatrix>;
/// A problem where constraints are declared first, and variables are then added dynamically.
/// See [`Problem<ColMatrix>`](Problem#impl).
pub type ColProblem = Problem<ColMatrix>;

mod matrix_col;
mod matrix_row;
mod options;
mod status;

/// A complete optimization problem.
/// Depending on the `MATRIX` type parameter, the problem will be built
/// constraint by constraint (with [ColProblem]), or
/// variable by variable (with [RowProblem])
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem<MATRIX = ColMatrix> {
    // columns
    colcost: Vec<f64>,
    collower: Vec<f64>,
    colupper: Vec<f64>,
    // rows
    rowlower: Vec<f64>,
    rowupper: Vec<f64>,
    matrix: MATRIX,
}

impl<MATRIX: Default> Problem<MATRIX>
where
    Problem<ColMatrix>: From<Problem<MATRIX>>,
{
    fn num_cols(&self) -> usize {
        self.colcost.len()
    }
    fn num_rows(&self) -> usize {
        self.rowlower.len()
    }

    fn add_row_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        let r = Row(self.num_rows().try_into().unwrap());
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.rowlower.push(low);
        self.rowupper.push(high);
        r
    }

    fn add_column_inner<N: Into<f64> + Copy, B: RangeBounds<N>>(
        &mut self,
        col_factor: f64,
        bounds: B,
    ) {
        self.colcost.push(col_factor);
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.collower.push(low);
        self.colupper.push(high);
    }

    /// Create a model based on this problem. Don't solve it yet.
    /// If the problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    pub fn optimise(self, sense: Sense) -> Model {
        let mut m = Model::new(self);
        m.set_sense(sense);
        m
    }

    /// Create a new problem instance
    pub fn new() -> Self {
        Self::default()
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) | Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> c_int {
    n.try_into().unwrap()
}

/// A model to solve
#[derive(Debug)]
pub struct Model {
    highs: HighsPtr,
}

/// A solved model
#[derive(Debug)]
pub struct SolvedModel {
    highs: HighsPtr,
}

/// Whether to maximize or minimize the objective function
#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    /// max
    Maximise = -1,
    /// min
    Minimise = 1,
}

impl Model {
    /// Set the optimization sense (minimize by default)
    pub fn set_sense(&mut self, sense: Sense) {
        let ret = unsafe { Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int) };
        assert_eq!(ret, 1, "changeObjectiveSense failed");
    }

    /// Create a Highs model to be optimized (but don't solve it yet).
    /// If the given problem is a [RowProblem], it will have to be converted to a [ColProblem] first,
    /// which takes an amount of time proportional to the size of the problem.
    pub fn new<P: Into<Problem<ColMatrix>>>(problem: P) -> Self {
        let mut highs = HighsPtr::default();
        let problem = problem.into();
        log::debug!(
            "Adding a problem with {} variables and {} constraints to HiGHS",
            problem.num_cols(),
            problem.num_rows()
        );
        unsafe {
            handle_status(Highs_passLp(
                highs.mut_ptr(),
                c(problem.num_cols()),
                c(problem.num_rows()),
                c(problem.matrix.avalue.len()),
                problem.colcost.as_ptr(),
                problem.collower.as_ptr(),
                problem.colupper.as_ptr(),
                problem.rowlower.as_ptr(),
                problem.rowupper.as_ptr(),
                problem.matrix.astart.as_ptr(),
                problem.matrix.aindex.as_ptr(),
                problem.matrix.avalue.as_ptr(),
            ));
        }
        Self { highs }
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        handle_status(unsafe { Highs_runQuiet(self.highs.mut_ptr()) })
    }

    /// Set a custom parameter on the model.
    /// For the list of available options and their documentation, see:
    /// <https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.html/>
    ///
    /// ```
    /// # use highs::ColProblem;
    /// # use highs::Sense::Maximise;
    /// let mut model = ColProblem::default().optimise(Maximise);
    /// model.set_option("presolve", "off"); // disable the presolver
    /// model.set_option("solver", "ipm"); // use the ipm solver
    /// model.set_option("time_limit", 30.0); // stop after 30 seconds
    /// model.set_option("parallel", "on"); // use multiple cores
    /// model.set_option("highs_min_threads", 4); // solve on 4 threads minimum
    /// ```
    pub fn set_option<STR: Into<Vec<u8>>, V: HighsOptionValue>(&mut self, option: STR, value: V) {
        let c_str = CString::new(option).expect("invalid option name");
        handle_status(unsafe { value.set_option(self.highs.mut_ptr(), c_str.as_ptr()) });
    }

    /// Find the optimal value for the problem
    pub fn solve(mut self) -> SolvedModel {
        unsafe {
            handle_status(Highs_run(self.highs.mut_ptr()));
        }
        SolvedModel { highs: self.highs }
    }
}

impl From<SolvedModel> for Model {
    fn from(solved: SolvedModel) -> Self {
        Self {
            highs: solved.highs,
        }
    }
}

#[derive(Debug)]
struct HighsPtr(*mut c_void);

impl Drop for HighsPtr {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.0) }
    }
}

impl Default for HighsPtr {
    fn default() -> Self {
        Self(unsafe { Highs_create() })
    }
}

impl HighsPtr {
    // To be used instead of unsafe_mut_ptr wherever possible
    #[allow(dead_code)]
    const fn ptr(&self) -> *const c_void {
        self.0
    }

    // Needed until https://github.com/ERGO-Code/HiGHS/issues/479 is fixed
    unsafe fn unsafe_mut_ptr(&self) -> *mut c_void {
        self.0
    }

    fn mut_ptr(&mut self) -> *mut c_void {
        self.0
    }
}

impl SolvedModel {
    /// The status of the solution. Should be Optimal if everything went well
    pub fn status(&self) -> HighsModelStatus {
        let model_status = unsafe { Highs_getModelStatus(self.highs.unsafe_mut_ptr(), 0) };
        HighsModelStatus::try_from(model_status).unwrap()
    }

    /// Get the solution to the problem
    pub fn get_solution(&self) -> Solution {
        let cols = self.num_cols();
        let rows = self.num_rows();
        let mut colvalue: Vec<f64> = vec![0.; cols];
        let mut coldual: Vec<f64> = vec![0.; cols];
        let mut rowvalue: Vec<f64> = vec![0.; rows];
        let mut rowdual: Vec<f64> = vec![0.; rows];

        // Get the primal and dual solution
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                colvalue.as_mut_ptr(),
                coldual.as_mut_ptr(),
                rowvalue.as_mut_ptr(),
                rowdual.as_mut_ptr(),
            );
        }

        Solution {
            colvalue,
            coldual,
            rowvalue,
            rowdual,
        }
    }

    /// Number of variables
    fn num_cols(&self) -> usize {
        let n = unsafe { Highs_getNumCols(self.highs.unsafe_mut_ptr()) };
        n.try_into().unwrap()
    }

    /// Number of constraints
    fn num_rows(&self) -> usize {
        let n = unsafe { Highs_getNumRows(self.highs.unsafe_mut_ptr()) };
        n.try_into().unwrap()
    }
}

/// Concrete values of the solution
#[derive(Clone, Debug)]
pub struct Solution {
    colvalue: Vec<f64>,
    coldual: Vec<f64>,
    rowvalue: Vec<f64>,
    rowdual: Vec<f64>,
}

impl Solution {
    /// The optimal values for each variables (in the order they were added)
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    /// The optimal values for each variables in the dual problem (in the order they were added)
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    /// The value of the constraint functions
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
    /// The value of the constraint functions in the dual problem
    pub fn dual_rows(&self) -> &[f64] {
        &self.rowdual
    }
}

fn handle_status(status: c_int) {
    match HighsStatus::try_from(status).unwrap() {
        HighsStatus::OK => {}
        HighsStatus::Warning => {
            log::warn!("Warning from HiGHS !");
        }
        HighsStatus::Error => {
            panic!("An error was encountered in HiGHS.");
        }
    }
}

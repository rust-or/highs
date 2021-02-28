use std::convert::{TryFrom, TryInto};
use std::ffi::c_void;
use std::ops::{Bound, RangeBounds};
use std::os::raw::c_int;

use highs_sys::*;

pub use status::{HighsModelStatus, HighsStatus};

mod status;

#[derive(Debug, Clone, Copy)]
pub struct Row(c_int);

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Problem {
    // columns
    colcost: Vec<f64>,
    collower: Vec<f64>,
    colupper: Vec<f64>,
    // rows
    rowlower: Vec<f64>,
    rowupper: Vec<f64>,
    // column-wise sparse constraints  matrix
    astart: Vec<c_int>,
    aindex: Vec<c_int>,
    avalue: Vec<f64>,
}

impl Problem {
    fn num_cols(&self) -> usize {
        self.colcost.len()
    }
    fn num_rows(&self) -> usize {
        self.rowlower.len()
    }
    fn num_non_zero(&self) -> usize {
        self.avalue.len()
    }

    /// Add a row (a constraint) to the problem.
    /// The bounds indicate the maximum allowed value for the linear combination of variables that will be in the constraint.
    /// The actual constraint definition happens when adding a variable to the problem with add_column.
    pub fn add_row<B: RangeBounds<f64>>(&mut self, bounds: B) -> Row {
        let r = Row(self.num_rows().try_into().unwrap());
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.rowlower.push(low);
        self.rowupper.push(high);
        r
    }

    /// Add a column (a variable) to the problem.
    /// col_factor represents the factor in front of the variable in the objective function.
    /// The row_factors argument defines how much this variable weights in each constraint.
    pub fn add_column<N: Into<f64> + Copy, B: RangeBounds<N>, I: IntoIterator<Item=(Row, f64)>>(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.colcost.push(col_factor);
        let low = bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY);
        let high = bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY);
        self.collower.push(low);
        self.colupper.push(high);
        self.astart.push(self.aindex.len().try_into().unwrap());
        for (row, factor) in row_factors.into_iter() {
            self.aindex.push(row.0);
            self.avalue.push(factor);
        }
    }
}

fn bound_value<N: Into<f64> + Copy>(b: Bound<&N>) -> Option<f64> {
    match b {
        Bound::Included(v) => Some((*v).into()),
        Bound::Excluded(v) => Some((*v).into()),
        Bound::Unbounded => None,
    }
}

fn c(n: usize) -> c_int {
    n.try_into().unwrap()
}


#[derive(Debug, Default)]
pub struct Model {
    highs: HighsPtr,
}

#[derive(Debug)]
pub struct SolvedModel {
    highs: HighsPtr,
}

#[repr(C)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Sense {
    Maximise = -1,
    Minimise = 1,
}

impl Model {
    /// Create a Highs model to be optimized (but don't solve it yet).
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_sense(&mut self, sense: Sense) {
        unsafe {
            Highs_changeObjectiveSense(self.highs.mut_ptr(), sense as c_int);
        }
    }

    pub fn set_problem(&mut self, problem: Problem) {
        unsafe {
            log::debug!(
                "Adding a problem with {} variables and {} constraints to HiGHS",
                problem.num_cols(),
                problem.num_rows()
            );
            handle_status(Highs_passLp(
                self.highs.mut_ptr(),
                c(problem.num_cols()),
                c(problem.num_rows()),
                c(problem.num_non_zero()),
                problem.colcost.as_ptr(),
                problem.collower.as_ptr(),
                problem.colupper.as_ptr(),
                problem.rowlower.as_ptr(),
                problem.rowupper.as_ptr(),
                problem.astart.as_ptr(),
                problem.aindex.as_ptr(),
                problem.avalue.as_ptr(),
            ));
        }
    }

    /// Prevents writing anything to the standard output when solving the model
    pub fn make_quiet(&mut self) {
        handle_status(unsafe { Highs_runQuiet(self.highs.mut_ptr()) })
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
        HighsPtr(unsafe { Highs_create() })
    }
}

impl HighsPtr {
    // To be used instead of unsafe_mut_ptr wherever possible
    #[allow(dead_code)]
    fn ptr(&self) -> *const c_void {
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
    pub fn status(&self) -> HighsModelStatus {
        let model_status = unsafe { Highs_getModelStatus(self.highs.unsafe_mut_ptr(), 0) };
        HighsModelStatus::try_from(model_status).unwrap()
    }
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

    fn num_cols(&self) -> usize {
        let n = unsafe { Highs_getNumCols(self.highs.unsafe_mut_ptr()) };
        n.try_into().unwrap()
    }

    fn num_rows(&self) -> usize {
        let n = unsafe { Highs_getNumRows(self.highs.unsafe_mut_ptr()) };
        n.try_into().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    colvalue: Vec<f64>,
    coldual: Vec<f64>,
    rowvalue: Vec<f64>,
    rowdual: Vec<f64>,
}

impl Solution {
    pub fn columns(&self) -> &[f64] {
        &self.colvalue
    }
    pub fn dual_columns(&self) -> &[f64] {
        &self.coldual
    }
    pub fn rows(&self) -> &[f64] {
        &self.rowvalue
    }
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
            panic!(
                "An error was encountered in HiGHS. This is probably a memory allocation error."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_problem_empty() {
        let mut model = Model::default();
        model.set_problem(Problem::default());
        let solved = model.solve();
        assert_eq!(solved.status(), HighsModelStatus::ModelEmpty);
    }

    #[test]
    fn solve_sum() {
        // max: x + 2y + z
        // under constraints:
        // c1: 3x +  y      <= 6
        // c2:       y + 2z <= 7
        let mut pb = Problem::default();
        let c1 = pb.add_row(..6.);
        let c2 = pb.add_row(..7.);
        // x
        pb.add_column(1., 0.., vec![(c1, 3.)]);
        // y
        pb.add_column(2., 0.., vec![(c1, 1.), (c2, 1.)]);
        // z
        pb.add_column(1., 0.., vec![(c2, 2.)]);
        let mut model = Model::default();
        model.set_problem(pb);
        model.set_sense(Sense::Maximise);

        let solved = model.solve();

        assert_eq!(solved.status(), HighsModelStatus::Optimal);

        let solution = solved.get_solution();
        // The expected solution is x=0  y=6  z=0.5
        assert_eq!(solution.columns(), vec![0., 6., 0.5]);
        // All the constraints are at their maximum
        assert_eq!(solution.rows(), vec![6., 7.]);
    }
}

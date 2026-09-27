//! col-oriented matrix to build a problem variable by variable
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::{Integrality, Problem};

/// Represents a constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Row(pub(crate) c_int);

impl Row {
    /// Gets the index of the row
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for Row {
    fn from(index: usize) -> Self {
        Row(crate::c(index))
    }
}

/// A constraint matrix to build column-by-column
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColMatrix {
    // column-wise sparse constraints  matrix
    pub(crate) astart: Vec<c_int>,
    pub(crate) aindex: Vec<c_int>,
    pub(crate) avalue: Vec<f64>,
}

/// To use these functions, you need to first add all your constraints, and then add variables
/// one by one using the [Row] objects.
impl Problem<ColMatrix> {
    /// Add a row (a constraint) to the problem.
    /// The concrete factors are added later, when creating columns.
    pub fn add_row<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        self.add_row_inner(bounds)
    }

    /// Add a continuous variable to the problem.
    ///  - `col_factor` represents the factor in front of the variable in the objective function.
    ///  - `bounds` represents the maximal and minimal allowed values of the variable.
    ///  - `row_factors` defines how much this variable weights in each constraint.
    ///
    /// ```
    /// use highs::{ColProblem, Sense};
    /// let mut pb = ColProblem::new();
    /// let constraint = pb.add_row(..=5); // adds a constraint that cannot take a value over 5
    /// // add a variable that has a coefficient 2 in the objective function, is >=0, and has a coefficient
    /// // 2 in the constraint
    /// pb.add_column(2., 0.., &[(constraint, 2.)]);
    /// ```
    pub fn add_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality(col_factor, bounds, row_factors, false);
    }

    /// Same as add_column, but forces the solution to contain an integer value for this variable.
    ///
    /// ```
    /// use highs::{ColProblem, Sense};
    /// let mut pb = ColProblem::new();
    /// let constraint = pb.add_row(..=5); // adds a constraint that cannot take a value over 5
    /// // add an integer variable that has a coefficient 2 in the objective function, is >=0, and has a coefficient
    /// // 2 in the constraint
    /// pb.add_integer_column(2., 0.., &[(constraint, 2.)]);
    /// ```
    pub fn add_integer_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality(col_factor, bounds, row_factors, true);
    }

    /// Same as add_column, but lets you define whether the new variable should be integral or continuous.
    #[inline]
    pub fn add_column_with_integrality<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
        is_integer: bool,
    ) {
        self.add_column_with_integrality_kind(col_factor, bounds, row_factors, is_integer.into());
    }

    /// Same as add_column, but lets you set the variable's [`Integrality`]
    /// (continuous, integer, semicontinuous, or semi-integer).
    #[inline]
    pub fn add_column_with_integrality_kind<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
        integrality: Integrality,
    ) {
        self.matrix
            .astart
            .push(self.matrix.aindex.len().try_into().unwrap());
        let iter = row_factors.into_iter();
        let (size, _) = iter.size_hint();
        self.matrix.aindex.reserve(size);
        self.matrix.avalue.reserve(size);
        for r in iter {
            let &(row, factor) = r.borrow();
            self.matrix.aindex.push(row.0);
            self.matrix.avalue.push(factor);
        }
        self.add_column_inner(col_factor, bounds, integrality);
    }

    /// Add a semicontinuous variable: its value is `0` or within `bounds`.
    ///
    /// The lower bound is the threshold below which (other than `0`) the
    /// variable may not lie. A finite upper bound is recommended by HiGHS,
    /// though an unbounded upper bound is also accepted.
    pub fn add_semi_continuous_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality_kind(
            col_factor,
            bounds,
            row_factors,
            Integrality::SemiContinuous,
        );
    }

    /// Add a semi-integer variable: its value is `0` or an integer within `bounds`.
    ///
    /// The lower bound is the threshold below which (other than `0`) the
    /// variable may not lie. A finite upper bound is recommended by HiGHS,
    /// though an unbounded upper bound is also accepted.
    pub fn add_semi_integer_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Row, f64)>,
        I: IntoIterator<Item = ITEM>,
    >(
        &mut self,
        col_factor: f64,
        bounds: B,
        row_factors: I,
    ) {
        self.add_column_with_integrality_kind(
            col_factor,
            bounds,
            row_factors,
            Integrality::SemiInteger,
        );
    }
}

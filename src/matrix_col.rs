//! col-oriented matrix to build a problem variable by variable
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::Problem;

/// Represents a constraint
#[derive(Debug, Clone, Copy)]
pub struct Row(pub(crate) c_int);

/// A constraint matrix to build column-by-column
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColMatrix {
    // column-wise sparse constraints  matrix
    pub(crate) astart: Vec<c_int>,
    pub(crate) aindex: Vec<c_int>,
    pub(crate) avalue: Vec<f64>,
}

impl Problem<ColMatrix> {
    /// Add a row (a constraint) to the problem.
    /// The concrete factors are added later, when adding columns
    pub fn add_row<N: Into<f64> + Copy, B: RangeBounds<N>>(&mut self, bounds: B) -> Row {
        self.add_row_inner(bounds)
    }

    /// Add a column (a variable) to the problem.
    /// col_factor represents the factor in front of the variable in the objective function.
    /// The row_factors argument defines how much this variable weights in each constraint.
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
        self.add_column_inner(col_factor, bounds);
    }
}

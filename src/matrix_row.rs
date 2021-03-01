//! row-oriented matrix to build a problem constraint by constraint
use std::borrow::Borrow;
use std::convert::TryInto;
use std::ops::RangeBounds;
use std::os::raw::c_int;

use crate::matrix_col::ColMatrix;
use crate::Problem;

/// Represents a variable
#[derive(Debug, Clone, Copy)]
pub struct Col(usize);

/// A complete optimization problem stored by row
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RowMatrix {
    /// column-wise sparse constraints  matrix
    /// Each element in the outer vector represents a column (a variable)
    columns: Vec<(Vec<c_int>, Vec<f64>)>,
}

impl Problem<RowMatrix> {
    /// add a variable to the problem
    pub fn add_column<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
    >(&mut self, col_factor: f64, bounds: B) -> Col {
        let col = Col(self.num_cols());
        self.add_column_inner(col_factor, bounds);
        self.matrix.columns.push((vec![], vec![]));
        col
    }

    /// add a constraint to the problem
    pub fn add_row<
        N: Into<f64> + Copy,
        B: RangeBounds<N>,
        ITEM: Borrow<(Col, f64)>,
        I: IntoIterator<Item=ITEM>
    >(
        &mut self,
        bounds: B,
        row_factors: I,
    ) {
        let num_rows: c_int = self.num_rows().try_into().unwrap();
        for r in row_factors {
            let &(col, factor) = r.borrow();
            let c = &mut self.matrix.columns[col.0];
            c.0.push(num_rows);
            c.1.push(factor);
        }
        self.add_row_inner(bounds);
    }
}

impl From<RowMatrix> for ColMatrix {
    fn from(m: RowMatrix) -> ColMatrix {
        let mut astart = Vec::with_capacity(m.columns.len());
        astart.push(0);
        let size: usize = m.columns.iter().map(|(v, _)| v.len()).sum();
        let mut aindex = Vec::with_capacity(size);
        let mut avalue = Vec::with_capacity(size);
        for (row_indices, factors) in m.columns {
            aindex.extend(row_indices);
            avalue.extend(factors);
            astart.push(aindex.len().try_into().unwrap());
        }
        ColMatrix {
            astart,
            aindex,
            avalue,
        }
    }
}

impl From<Problem<RowMatrix>> for Problem<ColMatrix> {
    fn from(pb: Problem<RowMatrix>) -> Problem<ColMatrix> {
        Problem {
            colcost: pb.colcost,
            collower: pb.collower,
            colupper: pb.colupper,
            rowlower: pb.rowlower,
            rowupper: pb.rowupper,
            matrix: pb.matrix.into(),
        }
    }
}
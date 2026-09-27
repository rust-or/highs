use std::convert::{TryFrom, TryInto};
use std::ptr::null_mut;

use highs_sys::*;

use crate::{
    try_handle_status, Col, HighsIisBoundStatus, HighsModelStatus, HighsPtr, HighsStatus, Model,
    Row, Solution, SolvedModel,
};

/// An infeasible subsystem (IIS) of a model:
/// bounds of some of its columns (variables) and rows (constraints) that together are infeasible.
/// Depending on the [`iis_strategy`](https://github.com/ERGO-Code/HiGHS/blob/master/docs/src/guide/advanced.md#irreducible-infeasibility-system-iis-detectionid-highs-iis) option used,
/// the subsystem could also be reduced to a minimal (irreducible) one.
///
/// Returned by [`SolvedModel::get_iis`] and [`Model::get_iis`](crate::Model::get_iis).
/// HiGHS checks that the subsystem is infeasible before returning it: when it cannot,
/// the `try_get_iis` methods return `Err(HighsStatus::Warning)`.
///
/// # Search strategy
///
/// How hard HiGHS searches depends on the `iis_strategy` option
/// (see [`Model::set_option`](crate::Model::set_option) and [`iis_strategy`](https://github.com/ERGO-Code/HiGHS/blob/master/docs/src/guide/advanced.md#irreducible-infeasibility-system-iis-detectionid-highs-iis)).
/// - `0` (the default) only finds trivial subsystems: inconsistent bounds, or a single row that
///   cannot be satisfied within the bounds of its columns. Otherwise, the IIS is empty.
/// - `2` searches the whole model, and may return a subsystem that is not irreducible (minimal).
/// - `6` (`2 + 4`) also reduces the subsystem to an irreducible one, at a higher cost.
///
/// HiGHS also flags some columns and rows as "maybe in conflict":
/// the members of a subsystem not proven irreducible, and every column and row when it finds no subsystem.
/// This flag is not exposed, and ignoring it is not unsound: the listed bounds are infeasible on
/// their own, and an empty IIS says nothing about which columns and rows are involved.
///
/// # Unsolved models
///
/// If the model was not solved, or its solve stopped early (e.g. on a time limit), HiGHS first
/// solves it again, with the `iis_time_limit` option (no limit by default) as time limit.
/// This changes the status and solution of the model.
///
/// # Integer, semi-continuous and semi-integer columns
///
/// HiGHS checks the subsystem on the continuous relaxation of the model, ignoring integrality.
/// An infeasibility that only comes from integrality is thus never returned:
/// - with the default `iis_strategy` (`0`), the IIS is empty;
/// - with a search of the whole model (e.g. `iis_strategy` `2` or `6`), the search keeps
///   integrality, but the check fails, and the `try_get_iis` methods return `Err(HighsStatus::Warning)`.
///
/// Semi-continuous and semi-integer columns are not relaxed: whatever the `iis_strategy`, HiGHS
/// keeps their `[lower, upper]` bounds, ignoring that they may also be 0.
/// An IIS involving such columns may thus not be infeasible in the original model.
#[derive(Clone, Debug)]
pub struct Iis {
    columns: Vec<(Col, HighsIisBoundStatus)>,
    rows: Vec<(Row, HighsIisBoundStatus)>,
}

impl Iis {
    /// Whether the IIS has no columns and no rows: HiGHS found no infeasible subsystem
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty() && self.rows.is_empty()
    }

    /// The columns (variables) in the IIS, with the bounds that take part in it
    pub fn columns(&self) -> &[(Col, HighsIisBoundStatus)] {
        &self.columns
    }

    /// The rows (constraints) in the IIS, with the bounds that take part in it
    pub fn rows(&self) -> &[(Row, HighsIisBoundStatus)] {
        &self.rows
    }

    /// Whether the given column is part of the IIS
    pub fn contains_column(&self, col: Col) -> bool {
        self.columns.iter().any(|&(c, _)| c == col)
    }

    /// Whether the given row is part of the IIS
    pub fn contains_row(&self, row: Row) -> bool {
        self.rows.iter().any(|&(r, _)| r == row)
    }
}

impl HighsPtr {
    /// Compute an IIS of the current model with `Highs_getIis`
    pub(crate) fn try_get_iis(&mut self) -> Result<Iis, HighsStatus> {
        let cols = self.num_cols()?;
        let rows = self.num_rows()?;
        let mut iis_num_col: HighsInt = 0;
        let mut iis_num_row: HighsInt = 0;
        let mut col_index: Vec<HighsInt> = vec![0; cols];
        let mut row_index: Vec<HighsInt> = vec![0; rows];
        let mut col_bound: Vec<HighsInt> = vec![0; cols];
        let mut row_bound: Vec<HighsInt> = vec![0; rows];

        // The per-column and per-row statuses are not requested: HiGHS (1.15) leaves them
        // unset when it cannot check the subsystem, but still copies them.
        let status = unsafe {
            highs_call!(Highs_getIis(
                self.mut_ptr(),
                &mut iis_num_col,
                &mut iis_num_row,
                col_index.as_mut_ptr(),
                row_index.as_mut_ptr(),
                col_bound.as_mut_ptr(),
                row_bound.as_mut_ptr(),
                null_mut(),
                null_mut()
            ))
        }?;
        // Notably returned when HiGHS cannot check that the subsystem is infeasible
        if status == HighsStatus::Warning {
            return Err(status);
        }

        let iis_num_col: usize = iis_num_col.try_into()?;
        let iis_num_row: usize = iis_num_row.try_into()?;
        col_index.truncate(iis_num_col);
        col_bound.truncate(iis_num_col);
        row_index.truncate(iis_num_row);
        row_bound.truncate(iis_num_row);

        let columns = col_index
            .into_iter()
            .zip(col_bound)
            .map(|(i, b)| Ok((Col(i.try_into()?), bound_status(b))))
            .collect::<Result<_, HighsStatus>>()?;
        let rows = row_index
            .into_iter()
            .zip(row_bound)
            .map(|(i, b)| (Row(i), bound_status(b)))
            .collect();
        Ok(Iis { columns, rows })
    }
}

fn bound_status(raw: HighsInt) -> HighsIisBoundStatus {
    HighsIisBoundStatus::try_from(raw)
        .unwrap_or_else(|e| panic!("HiGHS returned an unrecognized IIS bound status: {e:?}"))
}

impl SolvedModel {
    /// Compute an infeasible subsystem of the model: see [`Iis`].
    ///
    /// Meant to be called when [`SolvedModel::status`] is
    /// [`Infeasible`](crate::HighsModelStatus::Infeasible).
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value, or cannot check that the subsystem is infeasible.
    pub fn get_iis(&mut self) -> Iis {
        self.try_get_iis()
            .unwrap_or_else(|e| panic!("HiGHS error: {e:?}"))
    }

    /// Tries to compute an infeasible subsystem of the model: see [`Iis`].
    ///
    /// Returns `Err(HighsStatus::Warning)` if HiGHS emits a warning, notably when it cannot check
    /// that the subsystem is infeasible, or the error status value if HIGHS returned an error status.
    pub fn try_get_iis(&mut self) -> Result<Iis, HighsStatus> {
        self.highs.try_get_iis()
    }
}

impl Model {
    /// Compute an infeasible subsystem of the model: see [`Iis`].
    ///
    /// Meant to be called when [`Model::status`] is [`HighsModelStatus::Infeasible`].
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value, or cannot check that the subsystem is infeasible.
    pub fn get_iis(&mut self) -> Iis {
        self.try_get_iis()
            .unwrap_or_else(|e| panic!("HiGHS error: {e:?}"))
    }

    /// Tries to compute an infeasible subsystem of the model: see [`Iis`].
    ///
    /// Returns `Err(HighsStatus::Warning)` if HiGHS emits a warning, notably when it cannot check
    /// that the subsystem is infeasible, or the error status value if HIGHS returned an error status.
    pub fn try_get_iis(&mut self) -> Result<Iis, HighsStatus> {
        self.highs.try_get_iis()
    }

    /// Solve the model in place, and return an infeasible subsystem if it is infeasible
    /// (see [`Iis`], which may be empty), or its solution otherwise.
    ///
    /// The solution may not be optimal, e.g. if the solve reached a time limit:
    /// see [`Model::status`].
    ///
    /// # Panics
    ///
    /// If HIGHS returns an error status value, or cannot check that the subsystem is infeasible.
    pub fn solve_or_iis(&mut self) -> Result<Solution, Iis> {
        self.try_solve_or_iis()
            .unwrap_or_else(|e| panic!("HiGHS error: {e:?}"))
    }

    /// Solve the model in place, and return an infeasible subsystem if it is infeasible
    /// (see [`Iis`], which may be empty), or its solution otherwise.
    ///
    /// The solution may not be optimal, e.g. if the solve reached a time limit:
    /// see [`Model::status`].
    ///
    /// Returns `Err(HighsStatus::Warning)` if HiGHS emits a warning while computing the
    /// subsystem, notably when it cannot check that it is infeasible,
    /// or the error status value if HIGHS returned an error status.
    pub fn try_solve_or_iis(&mut self) -> Result<Result<Solution, Iis>, HighsStatus> {
        if self.try_solve_in_place()? == HighsModelStatus::Infeasible {
            Ok(Err(self.try_get_iis()?))
        } else {
            Ok(Ok(self.get_solution()))
        }
    }
}

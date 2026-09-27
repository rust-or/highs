use std::ops::Bound::{Included, Unbounded};

use highs::{Col, ColProblem, HighsModelStatus, Row, Sense};

#[test]
fn add_rows_and_columns() {
    let mut model = ColProblem::new().optimise(Sense::Maximise);
    let x = model.add_col(1., 0..=10, []);
    // The new columns and rows are numbered from num_cols and num_rows before the call
    let first_col = model.num_cols();
    model.add_columns([0..=10]);
    let y = Col::from(first_col);
    model.change_column_cost(y, 2.);
    model.add_row(..=7., [(x, 3.), (y, 1.)]);
    let first_row = model.num_rows();
    model.add_rows([(..=4., vec![(y, 1.)]), (..=100., vec![(x, 1.), (y, 1.)])]);
    assert_eq!(Row::from(first_row).index(), 1);
    assert_eq!((model.num_cols(), model.num_rows()), (2, 3));
    let solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    let solution = solved.get_solution();
    assert_eq!(solution.columns(), &[1., 4.]);
    assert_eq!(solution.rows(), &[7., 4., 5.]);
}

#[test]
fn add_rows_with_mixed_bounds_from_iterators() {
    let mut model = ColProblem::new().optimise(Sense::Maximise);
    model.add_columns((0..2).map(|_| (Included(0.), Unbounded)));
    let cols: Vec<Col> = (0..model.num_cols()).map(Col::from).collect();
    model.change_column_cost(cols[0], 1.);
    model.change_column_cost(cols[1], 1.);
    // x + y <= 4, x - y = 1, y >= 1
    let rows: [((_, _), &[f64]); 3] = [
        ((Unbounded, Included(4.)), &[1., 1.]),
        ((Included(1.), Included(1.)), &[1., -1.]),
        ((Included(1.), Unbounded), &[0., 1.]),
    ];
    model.add_rows(rows.iter().map(|&(bounds, coefs)| {
        let row = cols.iter().copied().zip(coefs.iter().copied());
        (bounds, row.filter(|&(_, coef)| coef != 0.))
    }));
    let solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    let solution = solved.get_solution();
    assert_eq!(solution.columns(), &[2.5, 1.5]);
    assert_eq!(solution.rows(), &[4., 1., 1.5]);
}

#[test]
fn add_empty_batches() {
    let mut model = ColProblem::new().optimise(Sense::Minimise);
    model.add_columns(std::iter::empty::<std::ops::RangeInclusive<f64>>());
    model.add_rows(std::iter::empty::<(
        std::ops::RangeInclusive<f64>,
        Vec<(Col, f64)>,
    )>());
    assert_eq!((model.num_cols(), model.num_rows()), (0, 0));
}

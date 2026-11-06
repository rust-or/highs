use highs::{ColProblem, HighsModelStatus, RowProblem, Sense};

#[test]
fn solve_in_place_then_resolve() {
    let mut pb = RowProblem::new();
    let x = pb.add_column(1., 0..);
    let y = pb.add_column(2., 0..);
    pb.add_row(..=6., [(x, 3.), (y, 1.)]);
    let mut model = pb.optimise(Sense::Maximise);
    assert_eq!(model.status(), HighsModelStatus::NotSet);

    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    assert_eq!(model.get_solution().columns(), &[0., 6.]);
    assert_eq!(model.objective_value(), 12.);

    model.change_column_bounds(x, 1..);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    assert_eq!(model.get_solution().columns(), &[1., 3.]);

    model.change_column_cost(y, 1.);
    model.solve_in_place();
    assert_eq!(model.objective_value(), 4.);
}

#[test]
fn add_row_and_col_then_solve_in_place() {
    let mut model = ColProblem::default().optimise(Sense::Minimise);
    let col = model.add_col(1., 1.., vec![]);
    model.add_row(..1., vec![(col, 1.)]);
    assert_eq!(model.num_nz(), 1);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    assert_eq!(model.get_solution().columns(), &[1.]);

    model.change_column_bounds(col, 2..);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Infeasible);
}

#[test]
fn column_getters() {
    let mut model = ColProblem::default().optimise(Sense::Minimise);
    let col = model.add_col(3., 1..5, vec![]);
    assert_eq!(model.get_column_bounds(col), (1., 5.));
    assert_eq!(model.get_column_cost(col), 3.);
    model.change_column_bounds(col, 0..);
    model.change_column_cost(col, -1.);
    assert_eq!(model.get_column_bounds(col), (0., f64::INFINITY));
    assert_eq!(model.get_column_cost(col), -1.);
}

#[test]
#[should_panic]
fn column_getter_out_of_range() {
    let mut other = ColProblem::default().optimise(Sense::Minimise);
    let col = other.add_col(1., 0..1, vec![]);
    let model = ColProblem::default().optimise(Sense::Minimise);
    model.get_column_cost(col);
}

#[test]
fn problem_column_getters() {
    let mut pb = RowProblem::new();
    let col = pb.add_column(3., 1..);
    assert_eq!(pb.get_column_bounds(col), (1., f64::INFINITY));
    assert_eq!(pb.get_column_cost(col), 3.);
    pb.change_column_bounds(col, 0..=4);
    pb.change_column_cost(col, 2.);
    assert_eq!(pb.get_column_bounds(col), (0., 4.));
    assert_eq!(pb.get_column_cost(col), 2.);
}

#[test]
fn clear_solver_then_resolve() {
    let mut pb = RowProblem::new();
    pb.add_column(1., 0..10);
    let mut model = pb.optimise(Sense::Maximise);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    model.clear_solver();
    assert_eq!(model.status(), HighsModelStatus::NotSet);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    assert_eq!(model.get_solution().columns(), &[10.]);
}

#[test]
fn clear_model() {
    let mut pb = RowProblem::new();
    let x = pb.add_column(1., 0..10);
    pb.add_row(..=5, [(x, 1.)]);
    let mut model = pb.optimise(Sense::Maximise);
    model.clear_model();
    assert_eq!((model.num_cols(), model.num_rows()), (0, 0));
}

#[test]
fn overwrite() {
    let mut pb = RowProblem::new();
    pb.add_column(1., 0..10);
    let mut model = pb.optimise(Sense::Maximise);
    model.solve_in_place();
    assert_eq!(model.get_solution().columns(), &[10.]);

    let mut other = ColProblem::new();
    let row = other.add_row(..=2.5);
    other.add_integer_column(1., 0.., [(row, 1.)]);
    model.overwrite(other);
    model.set_sense(Sense::Maximise);
    assert_eq!(model.solve_in_place(), HighsModelStatus::Optimal);
    assert_eq!(model.get_solution().columns(), &[2.]);
}

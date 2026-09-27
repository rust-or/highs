use highs::{Col, ColProblem, HessianFormat, HighsModelStatus, Model, RowProblem, Sense};

/// max x + 2y s.t. 3x + y <= 6, solved: x = 0, y = 6
fn solved_lp() -> (Model, Col) {
    let mut pb = RowProblem::new();
    let x = pb.add_column(1., 0..);
    let y = pb.add_column(2., 0..);
    pb.add_row(..=6., [(x, 3.), (y, 1.)]);
    let solved = pb.optimise(Sense::Maximise).solve();
    assert_eq!(solved.get_solution().columns(), &[0., 6.]);
    (Model::from(solved), x)
}

#[test]
fn clone_preserves_lp_and_basis() {
    let (model, _) = solved_lp();
    let mut clone = model.clone();
    assert_eq!((clone.num_cols(), clone.num_rows()), (2, 1));
    // Without presolve, only a copied basis lets simplex finish without iterating.
    clone.set_option("presolve", "off");
    let solved = clone.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    assert_eq!(solved.get_solution().columns(), &[0., 6.]);
    assert_eq!(solved.simplex_iteration_count(), 0);
}

#[test]
fn clone_is_independent() {
    let (model, x) = solved_lp();
    let mut clone = model.clone();
    clone.change_column_bounds(x, 1..);
    assert_eq!(clone.solve().objective_value(), 7.);
    assert_eq!(model.solve().get_solution().columns(), &[0., 6.]);
}

#[test]
fn clone_after_modification() {
    let (mut model, x) = solved_lp();
    model.change_column_bounds(x, 1..);
    let solved = model.clone().solve();
    assert_eq!(solved.get_solution().columns(), &[1., 3.]);
}

#[test]
fn clone_before_solve() {
    let mut pb = RowProblem::new();
    pb.add_column(1., 0..10);
    let model = pb.optimise(Sense::Maximise);
    let solved = model.clone().solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    assert_eq!(solved.get_solution().columns(), &[10.]);
}

#[test]
fn clone_preserves_integrality() {
    // max x + 2y  s.t.  x + y <= 3.5,  x - y >= 1,  y integer.
    // The LP relaxation optimum is (2.25, 1.25), the MIP one (2.5, 1).
    let mut pb = ColProblem::new();
    let c1 = pb.add_row(..3.5);
    let c2 = pb.add_row(1..);
    pb.add_column(1., 0.., [(c1, 1.), (c2, 1.)]);
    pb.add_integer_column(2., 0.., [(c1, 1.), (c2, -1.)]);
    let model = pb.optimise(Sense::Maximise);
    let solved = model.clone().solve();
    assert_eq!(solved.get_solution().columns(), &[2.5, 1.]);
}

#[test]
fn clone_preserves_hessian() {
    // min 0.5 x^2 - x  s.t.  0 <= x <= 10: optimum at x = 1 (x = 10 without the Hessian).
    let mut pb = RowProblem::new();
    pb.add_column(-1., 0..10);
    let mut model = pb.optimise(Sense::Minimise);
    model.pass_hessian(HessianFormat::Triangular, [[(0, 1.)]]);
    let solved = model.clone().solve();
    let x = solved.get_solution().columns()[0];
    assert!((x - 1.).abs() < 1e-6, "x = {x}");
}

use highs::{Col, Row, SolvedModel};
use highs::{ColProblem, HighsIisBoundStatus, HighsModelStatus, HighsStatus, Model, Sense};

use HighsIisBoundStatus::{Free, Lower, Upper};

fn model_with_iis_strategy(iis_strategy: Option<i32>) -> Model {
    let mut model = ColProblem::new().optimise(Sense::Minimise);
    if let Some(strategy) = iis_strategy {
        model.set_option("iis_strategy", strategy);
    }
    model
}

fn solve_infeasible(model: Model) -> SolvedModel {
    let solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Infeasible);
    solved
}

#[test]
fn solved_model_get_iis() {
    let mut model = model_with_iis_strategy(None);
    let x = model.add_col(0., 1.., []); // x >= 1
    let y = model.add_col(0., 1.., []); // y >= 1
    let z = model.add_col(0., 1.., []); // z >= 1
    let r0 = model.add_row(..=1.1, [(x, 1.), (z, 1.)]); // x + z <= 1.1: infeasible with x, z >= 1
    let r1 = model.add_row(5.., [(y, 1.)]); // y >= 5: independent

    let iis = solve_infeasible(model).get_iis();
    assert_eq!(iis.columns(), &[(x, Lower), (z, Lower)]);
    assert_eq!(iis.rows(), &[(r0, Upper)]);
    assert!(!iis.is_empty());
    assert!(iis.contains_column(x));
    assert!(!iis.contains_column(y));
    assert!(iis.contains_column(z));
    assert!(iis.contains_row(r0));
    assert!(!iis.contains_row(r1));
}

/// x + y >= 3, x <= 1, y <= 1: infeasible, but not because of a single row
fn non_trivially_infeasible(iis_strategy: Option<i32>) -> (SolvedModel, [Col; 2], [Row; 3]) {
    let mut model = model_with_iis_strategy(iis_strategy);
    let x = model.add_col(0., 0..10, []);
    let y = model.add_col(0., 0..10, []);
    let r0 = model.add_row(3.., [(x, 1.), (y, 1.)]);
    let r1 = model.add_row(..=1, [(x, 1.)]);
    let r2 = model.add_row(..=1, [(y, 1.)]);
    (solve_infeasible(model), [x, y], [r0, r1, r2])
}

#[test]
fn default_iis_strategy_only_finds_trivial_subsystems() {
    let (mut solved, _, _) = non_trivially_infeasible(None);
    assert!(solved.get_iis().is_empty());
}

#[test]
fn irreducible_iis_strategy() {
    let (mut solved, [x, y], [r0, r1, r2]) = non_trivially_infeasible(Some(2 + 4));
    let iis = solved.get_iis();
    assert_eq!(iis.columns(), &[(x, Free), (y, Free)]);
    assert_eq!(iis.rows(), &[(r0, Lower), (r1, Upper), (r2, Upper)]);
}

/// 2y = 1 with y integer: the relaxation (y = 0.5) is feasible
fn infeasible_because_of_integrality(iis_strategy: Option<i32>) -> SolvedModel {
    let mut model = model_with_iis_strategy(iis_strategy);
    let y = model.add_integer_column(0., 0..10, []);
    model.add_row(1.0..=1.0, [(y, 2.)]);
    solve_infeasible(model)
}

#[test]
fn iis_of_mip_infeasible_because_of_integrality_is_empty() {
    assert!(infeasible_because_of_integrality(None).get_iis().is_empty());
}

#[test]
fn iis_of_mip_infeasible_because_of_integrality_cannot_be_checked() {
    let mut solved = infeasible_because_of_integrality(Some(2));
    assert_eq!(solved.try_get_iis().err(), Some(HighsStatus::Warning));
}

#[test]
fn iis_of_mip_with_infeasible_relaxation() {
    // x + y <= 1 with x >= 1 and y >= 1, y integer.
    let mut model = model_with_iis_strategy(None);
    let x = model.add_col(0., 1.., []);
    let y = model.add_integer_column(0., 1..5, []);
    let row = model.add_row(..=1.0, [(x, 1.), (y, 1.)]);
    let iis = solve_infeasible(model).get_iis();
    assert_eq!(iis.columns(), &[(x, Lower), (y, Lower)]);
    assert_eq!(iis.rows(), &[(row, Upper)]);
}

#[test]
fn iis_of_feasible_semi_continuous_model_is_empty() {
    // x in {0} U [5, 10] with x <= 1: feasible with x = 0.
    let mut model = model_with_iis_strategy(None);
    let x = model.add_semi_continuous_column(0., 5..10, []);
    model.add_row(..=1.0, [(x, 1.)]);
    let mut solved = model.solve();
    assert_eq!(solved.status(), HighsModelStatus::Optimal);
    assert!(solved.get_iis().is_empty());
}

#[test]
fn model_solve_or_iis() {
    let mut model = model_with_iis_strategy(None);
    let col = model.add_col(1., 1.., []);
    let r0 = model.add_row(..=2., [(col, 1.)]);
    let solution = model.solve_or_iis().expect("feasible");
    assert_eq!(solution.columns(), &[1.]);

    let r1 = model.add_row(..0.5, [(col, 1.)]); // col >= 1 but row <= 0.5
    let iis = model.solve_or_iis().expect_err("infeasible");
    assert!(iis.contains_column(col));
    assert!(iis.contains_row(r1));
    assert!(!iis.contains_row(r0));
    assert_eq!(model.status(), HighsModelStatus::Infeasible);
    assert!(!model.get_iis().is_empty());
}

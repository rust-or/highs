# highs

[![highs docs badge](https://docs.rs/highs/badge.svg)](https://docs.rs/highs)

Safe rust bindings to the Highs MILP Solver.

## Usage example

```rust

fn main() {
    // max: x + 2y + z
    // under constraints:
    // c1: 3x +  y      <= 6
    // c2:       y + 2z <= 7
    let mut pb = Problem::default();
    let c1 = pb.add_row(..6.);
    let c2 = pb.add_row(..7.);
    // x
    pb.add_column(
        1., // the coefficient in front of x in the objective function
        0.., // x is in the range 0, +∞ 
        &[ // Giving a slice, but anything that can be iterated over will do
            (c1, 3.) // The coefficient in front of x in the c1 constraint
        ]
    );
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
```

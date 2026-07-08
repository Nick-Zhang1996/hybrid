"""Tests and tuning plots for air traffic control costs."""
from rd3g.games.air_traffic_control_casadi import (
    AirTrafficControlCasadi,
    AirTrafficControlCasadiConfig,
    create_multi_valley_fun,
)
import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def _eval_scalar_fun(fun, values):
    return np.asarray([float(fun(float(value))) for value in values])


def _eval_stage_cost(game, state):
    gc = game.config
    x = cas.SX.sym('x', gc.n, gc.N)
    u = cas.SX.sym('u', gc.m, 1)
    i_onehot = cas.SX.sym('i_onehot', gc.N, 1)
    stage_cost = cas.Function(
        'stage_cost',
        [x, u, i_onehot, gc.get_int_param_sx(), gc.get_double_param_sx()],
        [game.J(x, u, i_onehot)],
    )
    return float(stage_cost(
        np.asarray(state, dtype=float).reshape(gc.n, gc.N),
        cas.DM.zeros(gc.m, 1),
        cas.DM.ones(gc.N, 1),
        gc.get_int_param_np(),
        gc.get_double_param_np(),
    ))


def test_multi_valley_fun_returns_casadi_function():
    fun = create_multi_valley_fun([0.0, 300.0])

    assert isinstance(fun, cas.Function)


def test_multi_valley_fun_has_one_valley_per_target():
    targets = [0.0, 300.0]
    fun = create_multi_valley_fun(targets, width_fraction=0.12, beta=30.0)

    target_costs = _eval_scalar_fun(fun, targets)
    left_neighbor_costs = _eval_scalar_fun(fun, [-10.0, 290.0])
    right_neighbor_costs = _eval_scalar_fun(fun, [10.0, 310.0])
    off_target_costs = _eval_scalar_fun(fun, [-90.0, 150.0, 390.0])

    assert np.all(target_costs < left_neighbor_costs)
    assert np.all(target_costs < right_neighbor_costs)
    assert np.all(off_target_costs > target_costs.max())


def test_multi_valley_fun_is_unbounded_away_from_targets():
    targets = [1.0, 3.0]
    fun = create_multi_valley_fun(targets, width_fraction=0.18, beta=50.0)
    val = cas.SX.sym('val')
    gradient_fun = cas.Function('multi_valley_gradient', [val],
                                [cas.gradient(fun(val), val)])

    near_cost = float(fun(4.0))
    far_cost = float(fun(6.0))
    farther_cost = float(fun(8.0))
    far_gradient = float(gradient_fun(6.0))
    farther_gradient = float(gradient_fun(8.0))

    assert near_cost < far_cost
    assert far_cost < farther_cost
    assert 0.0 < far_gradient < farther_gradient


def test_multi_valley_fun_is_scale_agnostic():
    targets = [1.0, 3.0]
    scaled_targets = [10.0, 30.0]
    xs = np.linspace(-1.0, 5.0, 101)

    fun = create_multi_valley_fun(targets, width_fraction=0.12, beta=30.0)
    scaled_fun = create_multi_valley_fun(scaled_targets, width_fraction=0.12, beta=30.0)

    y = _eval_scalar_fun(fun, xs)
    y_scaled = _eval_scalar_fun(scaled_fun, 10.0 * xs)

    np.testing.assert_allclose(y, y_scaled, rtol=1e-12, atol=1e-12)


def test_visualize_multi_valley_fun_for_tuning(tmp_path):
    targets = [1.0, 10.0]
    max_val = np.max(targets)
    min_val = np.min(targets)
    span = max_val - min_val
    xs = np.linspace(min_val - span, max_val + span, 800)
    parameter_sets = [
        {'width_fraction': 0.08, 'beta': 20.0},
        {'width_fraction': 0.12, 'beta': 30.0},
        {'width_fraction': 0.18, 'beta': 50.0},
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    for params in parameter_sets:
        fun = create_multi_valley_fun(targets, **params)
        ys = _eval_scalar_fun(fun, xs)
        label = f"width={params['width_fraction']}, beta={params['beta']}"
        ax.plot(xs, ys, label=label)

    for target in targets:
        ax.axvline(target, color='black', linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xlabel('val')
    ax.set_ylabel('cost')
    ax.set_title('multi-valley cost tuning')
    ax.legend()
    ax.grid(True, alpha=0.25)

    output_path = tmp_path / 'multi_valley_fun_tuning.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    assert output_path.exists()
    print(f'multi-valley tuning plot: {output_path}')


def test_atc_stage_cost_penalizes_heading_and_lateral_deviation():
    config = AirTrafficControlCasadiConfig(
        N=1,
        x0=np.array([[0.0], [0.0], [80.0], [0.0]], dtype=float),
    )
    game = AirTrafficControlCasadi(config)

    aligned_cost = _eval_stage_cost(game, [0.0, 0.0, 80.0, 0.0])
    heading_off_cost = _eval_stage_cost(game, [0.0, 0.0, 80.0, np.deg2rad(45.0)])
    lateral_off_cost = _eval_stage_cost(game, [0.0, 100.0, 80.0, 0.0])

    assert aligned_cost < heading_off_cost
    assert aligned_cost < lateral_off_cost

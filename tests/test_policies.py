from coverage_planner.env import Agent, GridWorld
from coverage_planner.objective import Coverage
from coverage_planner.policies.tools import build_path_model, enumerate_paths
from coverage_planner.policies.greedy import (
    full_horizon_greedy_solve,
    rolling_horizon_greedy_solve,
)

GRID_SIZE = 5
STEPS = 4


def make_model(num_agents, size=GRID_SIZE, steps=STEPS, start=(0, 0)):
    Agent._id_counter = 0
    agents = [Agent(*start) for _ in range(num_agents)]
    grid = GridWorld(size)
    grid.init_agent(agents)
    return build_path_model(grid, Coverage, steps, [a.id for a in agents])


def in_bounds(path, size=GRID_SIZE):
    return all(0 <= r < size and 0 <= c < size for r, c in path)


# --- path enumeration ---

def test_enumerated_paths_have_correct_length():
    grid = GridWorld(4)
    paths = enumerate_paths(grid, steps=3, start_row=1, start_col=2)
    assert paths
    # start position + one coordinate per step
    assert all(len(p) == 4 for p in paths)


def test_enumerated_paths_stay_inside_grid():
    grid = GridWorld(4)
    paths = enumerate_paths(grid, steps=3, start_row=0, start_col=0)
    assert all(in_bounds(p, size=4) for p in paths)


def test_enumerated_paths_start_at_start_position():
    grid = GridWorld(4)
    paths = enumerate_paths(grid, steps=2, start_row=3, start_col=1)
    assert all(p[0] == (3, 1) for p in paths)


# --- solver path validity ---

def test_solver_paths_have_correct_length_and_stay_in_bounds():
    model = make_model(3)
    results = [
        full_horizon_greedy_solve(model),
        rolling_horizon_greedy_solve(model, chunksize=3),  # 4 = 3 + 1 remainder
    ]
    for result in results:
        assert len(result.paths_by_agent) == 3
        for path in result.paths_by_agent.values():
            assert len(path) == STEPS + 1
            assert in_bounds(path)


# --- score properties ---

def test_score_nondecreasing_when_adding_agents():
    scores = [
        full_horizon_greedy_solve(make_model(n)).score
        for n in range(1, 5)
    ]
    assert scores == sorted(scores)


def test_score_bounded_by_grid_and_reachable_cells():
    model = make_model(2)
    result = full_horizon_greedy_solve(model)
    assert 0 < result.score <= GRID_SIZE ** 2
    assert result.score <= 2 * (STEPS + 1)  # agents * cells per path


# --- rolling-horizon consistency ---

def test_chunksize_equal_to_steps_matches_full_horizon():
    model = make_model(3)
    full = full_horizon_greedy_solve(model)
    rolling = rolling_horizon_greedy_solve(model, chunksize=STEPS)
    assert rolling.score == full.score
    assert rolling.chosen_path_ids == full.chosen_path_ids


def test_rolling_horizon_never_beats_full_horizon_here():
    # Not a theorem in general, but for these small same-start cases the
    # chunked solver should not out-score the full-horizon plan.
    model = make_model(3)
    full = full_horizon_greedy_solve(model)
    for chunk in range(1, STEPS + 1):
        rolling = rolling_horizon_greedy_solve(model, chunksize=chunk)
        assert rolling.score <= full.score

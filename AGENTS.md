# UPF_RPE Project Guidelines

## Package Layout

- **Keep `Code/` as main package** (do not convert to `src/`)
- Module organization:
  - `Code/PFilter/` - Particle filter implementations (UPF, UKF)
  - `Code/BaseLines/` - Comparison algorithms (NLS, QCQP, Algebraic)
  - `Code/Simulation/` - Robot models, trajectories, movement
  - `Code/UtilityCode/` - Math utilities (SE(2,3), transformations)
  - `Code/DataLoggers/` - Logging for each algorithm
  - `Code/test/` - Test modules

## Testing Standards

- **Keep unittest framework** (existing codebase pattern)
- Use `_ut.py` suffix for fast unit tests
- Use `test_*.py` for integration/simulation tests
- Plot tests may call `plt.show()` but should be marked with:
  ```python
  @unittest.skipUnless(plot_bool, 'Plot test')
  ```
- Centralize test fixtures in `Code/test/fixtures.py`
- Use deterministic seeds for reproducible simulations

## Dependencies & Environment

- **Add `pyproject.toml`** for dependency management (in addition to `requirements.txt`)
- Minimum Python: 3.11 (verify during setup)
- Core dependencies:
  - numpy, scipy, filterpy (essential filtering)
  - cvxpy, gurobipy (QCQP solver - requires Gurobi license)
  - matplotlib (visualization)
  - torch (optional, for deep learning extensions)
- Virtual env location: `.venv/` project-local
- Install missing dependencies via pip before running tests

## Git Workflow

Inherit workspace-level rules from `/workspace/AGENTS.md`:
- Protected main/master branches
- Create new branch before changes (if on main/master)
- Push after completing changes unless explicitly told not to
- Use imperative commit messages: "Codex: Add feature X"

## File Hygiene

Inherit workspace-level rules:
- Files under `/workspace` should have owner `yuri`, group `gituser`
- Permissions: `775` (not `755`)
- Run `chown -R yuri:gituser .` after git push
- Do not delete experimental data in `Data/` without explicit permission

## Algorithm-Specific Guidelines

- **QCQP requires Gurobi license** (document clearly in code comments)
- UPF parameters are configured in `Code/PFilter/ConnectedAgentClass.py`
  - UKF parameters: kappa, alpha, beta
  - Particle filter: number of particles, resample factor
- Simulation parameters should be in `test_cases/*/parameters.py` where possible
- Always use same random seeds for fair algorithm comparison

## Data Management

- Experimental rosbags with camera images not committed (email yuri.durodie@vub.be)
- Simulation data: `Data/Simulations/`
- Results from experiments/simulations: `Results/` with timestamped subdirectories
- Never delete original experimental/simulation data without explicit permission

## Documentation Standards

- Public classes/methods: NumPy-style docstrings
- Include mathematical formulas in comments for key algorithms
- Update `README.md` when adding new algorithms or major features
- Document parameter meanings and units clearly

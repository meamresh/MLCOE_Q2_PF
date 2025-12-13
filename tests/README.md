# Tests

This directory contains unit tests and integration tests for the MLCOE_Q2_PF project.

## Test Structure

- `test_models.py`: Unit tests for state-space models (LGSSM)
- `test_filters.py`: Unit tests for Kalman filter implementation
- `test_metrics.py`: Unit tests for accuracy and stability metrics
- `test_data_generators.py`: Unit tests for data generation and YAML parsing
- `test_integration.py`: Integration tests for the full pipeline

## Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Or using unittest:
```bash
python -m unittest discover tests -v
```

Run a specific test file:
```bash
python -m unittest tests.test_models -v
```

Run a specific test class:
```bash
python -m unittest tests.test_models.TestLGSSM -v
```

Run a specific test method:
```bash
python -m unittest tests.test_models.TestLGSSM.test_model_creation -v
```

## Test Coverage

The tests cover:
- Model creation and configuration
- Data generation and sampling
- Kalman filter predict/update steps
- Riccati vs Joseph covariance updates
- Accuracy metrics (RMSE, MAE, NEES)
- Stability metrics (condition numbers, symmetry, positive definiteness)
- Full pipeline integration
- Filter consistency checks


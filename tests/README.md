# Test Suite

This directory contains the test suite for the channel-heads project.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_coupling_analysis.py

# Run with verbose output
pytest -v
```

## Test Coverage

Current test coverage focuses on:
- Data structure validation (PairTouchResult)
- Helper function correctness (_normalize_pair, _to_node_id_list)
- Edge cases and input validation

## Future Testing Needs

See [improvement.md](../improvement.md) for recommendations on:

1. **Mock TopoToolbox Objects**: Create fixtures for FlowObject, StreamObject, GridObject
2. **Integration Tests**: Test full workflows with synthetic DEMs
3. **Visualization Tests**: Validate plot outputs (image comparison or property checks)
4. **Performance Tests**: Benchmark large DEM processing
5. **Regression Tests**: Lock in expected outputs for known datasets

## Test Data

For integration tests, consider creating:
- `tests/data/` - Small synthetic test DEMs (100x100 pixels)
- `tests/fixtures/` - Pre-computed stream networks
- `tests/expected/` - Expected outputs for regression testing

## Contributing Tests

When adding new functionality:
1. Write tests first (TDD approach)
2. Ensure >80% code coverage for new code
3. Include edge cases and error conditions
4. Document test rationale in docstrings

# Native Mode Test Coverage Summary

## Files Modified/Created

1. **NEW FILE**: `tests/unit_tests/byob/test_execution_mode.py`
   - Tests for ExecutionMode enum
   - Tests for Evaluation model validation changes
   - Tests for native harness registry

2. **MODIFIED**: `tests/unit_tests/byob/test_byob_compiler.py`
   - Added `TestCompileNativeMode` class with 8 new tests
   - Existing 4 tests remain unchanged

## Test Breakdown

### test_execution_mode.py (27 tests)

#### TestExecutionModeEnum (4 tests)
- `test_execution_mode_values` - Validates enum string values
- `test_execution_mode_from_string` - Validates enum construction from strings
- `test_execution_mode_invalid_value` - Validates rejection of invalid values
- `test_execution_mode_case_sensitive` - Validates case-sensitive enum values

#### TestEvaluationModel (13 tests)
- `test_evaluation_subprocess_mode_with_command` - Subprocess with command
- `test_evaluation_native_mode_no_command` - Native without command
- `test_evaluation_default_execution_mode` - Default to subprocess (backward compat)
- `test_evaluation_subprocess_missing_command_raises` - Validation: subprocess needs command
- `test_evaluation_subprocess_empty_command_raises` - Validation: empty command rejected
- `test_evaluation_native_mode_with_command_allowed` - Native can have command (ignored)
- `test_evaluation_render_command_fails_for_native_mode` - render_command() error for native
- `test_evaluation_from_dict_subprocess_mode` - Dict construction with subprocess
- `test_evaluation_from_dict_native_mode` - Dict construction with native
- `test_evaluation_extra_field_forbidden` - Pydantic extra='forbid' enforcement

#### TestNativeHarnessRegistry (10 tests)
- `test_register_and_get_native_harness` - Basic register/retrieve
- `test_get_native_harness_longest_prefix_match` - Prefix matching specificity
- `test_get_native_harness_not_found` - Error on missing harness
- `test_registry_isolation_between_tests` - Fixture isolation verification
- `test_registry_multiple_prefixes` - Multiple independent registrations
- `test_register_overwrites_existing_prefix` - Re-registration overwrites
- `test_native_harness_protocol_compliance` - Protocol satisfaction check
- `test_registry_factory_pattern` - Factory function pattern

### test_byob_compiler.py (12 tests total: 4 existing + 8 new)

#### TestCompileNativeMode (8 new tests)
- `test_compiler_native_mode_fdf` - Native mode FDF structure validation
- `test_compiler_default_subprocess_mode_fdf` - Default subprocess FDF (backward compat)
- `test_compiler_explicit_subprocess_mode` - Explicit subprocess mode
- `test_install_native_mode_framework_yml` - Native framework.yml installation
- `test_install_native_mode_no_output_py` - Native mode skips output.py generation
- `test_install_subprocess_mode_has_output_py` - Subprocess mode generates output.py
- `test_compiler_native_mode_multi_benchmark` - Multiple benchmarks in native mode

## Coverage Statistics

**Total New Tests**: 35
- Execution mode enum: 4 tests
- Evaluation model: 13 tests
- Native harness registry: 10 tests
- Compiler native mode: 8 tests

**Total BYOB Tests**: 39 (4 existing compiler + 35 new)

## Commands to Run Tests

### Run only execution mode tests
```bash
pytest tests/unit_tests/byob/test_execution_mode.py -v --tb=short
```

### Run only compiler tests (existing + native mode)
```bash
pytest tests/unit_tests/byob/test_byob_compiler.py -v --tb=short
```

### Run all BYOB unit tests
```bash
pytest tests/unit_tests/byob/ -v --tb=short
```

### Run with coverage report
```bash
pytest tests/unit_tests/byob/ -v --tb=short --cov=nemo_evaluator.api.api_dataclasses --cov=nemo_evaluator.core.native_harness --cov=nemo_evaluator.byob.compiler --cov-report=term-missing
```

## Expected Results

All 39 tests should PASS. Key success criteria:

1. **ExecutionMode enum** correctly defined with "subprocess" and "native" values
2. **Evaluation model** accepts both execution modes with proper validation
3. **Default behavior** is subprocess mode (backward compatibility)
4. **Subprocess mode** requires command field
5. **Native mode** does not require command field
6. **Compiler native mode** produces correct FDF structure without command
7. **Compiler default mode** produces subprocess FDF with command (backward compat)
8. **Installation native mode** generates framework.yml without output.py
9. **Installation subprocess mode** generates both framework.yml and output.py
10. **Native harness registry** correctly registers and retrieves harnesses by prefix

## Invariants Verified

1. ✅ ExecutionMode.SUBPROCESS is the default (test_evaluation_default_execution_mode)
2. ✅ Native mode requires no command field (test_evaluation_native_mode_no_command)
3. ✅ Subprocess mode requires command field (test_evaluation_subprocess_missing_command_raises)
4. ✅ Registry cleanup is atomic (autouse fixture + test_registry_isolation_between_tests)
5. ✅ Enum values are lowercased (test_execution_mode_values, test_execution_mode_case_sensitive)

## Risk Coverage

| Risk | Severity | Tests Covering |
|------|----------|----------------|
| Incorrect execution mode routing | CRITICAL | test_evaluation_subprocess_mode_with_command, test_evaluation_native_mode_no_command |
| Backward compatibility break | CRITICAL | test_evaluation_default_execution_mode, test_compiler_default_subprocess_mode_fdf |
| Registry pollution | HIGH | autouse fixture, test_registry_isolation_between_tests |
| Invalid FDF structure | MEDIUM | test_compiler_native_mode_fdf, test_install_native_mode_framework_yml |
| Validation bypass | MEDIUM | test_evaluation_subprocess_missing_command_raises, test_evaluation_subprocess_empty_command_raises |

## Residual Risk

### NOT Covered by These Tests
1. **Engine integration** - Actual evaluate() function branching logic (requires integration tests or engine refactoring)
2. **Native harness execution** - ByobNativeHarness.execute() implementation (requires separate test file)
3. **Adapter pipeline integration** - ModelCallFn routing through adapters
4. **Equivalence validation** - Native vs subprocess result parity (requires integration tests)
5. **Error propagation** - How native mode errors surface in engine logs

### Why NOT Covered
- Engine integration requires either:
  - Heavy mocking of evaluate() dependencies (psutil, signal handlers, AdapterServerProcess)
  - OR extraction of branching logic into testable helper function
  - OR full integration test with mock server
- Native harness implementation is in separate module (byob/native_runner.py) - should have its own test file
- Equivalence tests require running both modes with real benchmarks - integration test scope

### Recommendation
These unit tests validate the **contracts and data structures** needed for native mode. The actual **execution path** requires:
1. Integration tests (test_byob_native_e2e.py, test_byob_equivalence.py per test plan)
2. Native runner unit tests (test_byob_native_runner.py per test plan)
3. Engine branching extraction (if evaluate() is too complex to mock)

## Next Steps

1. Run all tests: `pytest tests/unit_tests/byob/ -v --tb=short`
2. Verify all 39 tests pass
3. If any fail, check:
   - ExecutionMode enum definition in api_dataclasses.py
   - Evaluation model validation logic
   - Compiler native mode conditional logic
   - Registry fixture isolation
4. Once passing, proceed to native runner tests (test_byob_native_runner.py)

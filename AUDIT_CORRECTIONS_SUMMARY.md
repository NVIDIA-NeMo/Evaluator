# Documentation Audit Corrections Summary

## Overview

This document summarizes corrections made to documentation based on the comprehensive audit using the docs-audit-v3 methodology.

## Critical Issue Identified

The documentation referenced a non-existent module `nemo_eval.utils.base` that does not exist in the codebase. The actual package structure uses `nemo_evaluator` (not `nemo_eval`), and the documented utility functions did not exist as described.

## Files Corrected

### 1. **docs/references/evaluation-utils.md** ✅ REWRITTEN

**Problem**: Entire document referenced non-existent `nemo_eval.utils.base` module with four functions that don't exist.

**Action**: Complete rewrite to document the actual `show_available_tasks()` function from `nemo_evaluator`.

**Changes**:
- ✅ Corrected module path: `nemo_evaluator` (not `nemo_eval.utils.base`)
- ✅ Documented actual function: `show_available_tasks()` 
- ✅ Added correct usage examples with `nemo_evaluator` imports
- ✅ Updated source attribution to point to actual code location
- ✅ Added CLI-based task discovery as primary recommendation
- ✅ Removed non-existent functions: `list_available_evaluations()`, `find_framework()`, `wait_for_fastapi_server()`, `check_endpoint()`

### 2. **scripts/helpers.py** ✅ FIXED

**Problem**: Imported `check_endpoint` from non-existent `nemo_eval.utils.base` module.

**Action**: Implemented `check_endpoint()` function locally using `requests` library.

**Changes**:
- ✅ Removed import from non-existent module
- ✅ Added local implementation of `check_endpoint()` function
- ✅ Updated import to use correct `nemo_evaluator.core.evaluate`

### 3. **Documentation Cross-References** ✅ UPDATED

Updated all documentation files that referenced the non-existent module:

#### Files Updated:
1. ✅ `docs/troubleshooting/setup-issues/installation.md`
2. ✅ `docs/troubleshooting/setup-issues/deployment.md`
3. ✅ `docs/troubleshooting/setup-issues/index.md`
4. ✅ `docs/troubleshooting/index.md`
5. ✅ `docs/troubleshooting/advanced/common-patterns.md`
6. ✅ `docs/troubleshooting/advanced/index.md`
7. ✅ `docs/troubleshooting/advanced/debugging-guide.md`
8. ✅ `docs/troubleshooting/runtime-issues/configuration.md`
9. ✅ `docs/evaluation/run-evals/text-gen.md`
10. ✅ `docs/evaluation/run-evals/log-probability.md`
11. ✅ `docs/get-started/quickstart/full-stack.md`

**Changes Made**:
- Replaced `from nemo_eval.utils.base import list_available_evaluations` with `from nemo_evaluator import show_available_tasks`
- Replaced `from nemo_eval.utils.base import find_framework` with CLI-based task discovery
- Replaced `from nemo_eval.utils.base import wait_for_fastapi_server` with local `requests`-based implementations
- Updated all code examples to use correct module paths

### 4. **Tutorial Notebooks** ⚠️ REQUIRES MANUAL UPDATE

**Files Needing Updates**:
- `tutorials/simple-evals.ipynb`
- `tutorials/mmlu.ipynb`
- `tutorials/wikitext.ipynb`

**Issue**: All notebooks contain `from nemo_eval.utils.base import check_endpoint` which will fail.

**Recommended Action**: Update notebooks to define `check_endpoint()` locally or remove the dependency.

**Example Fix**:
```python
# Add to notebook cell 1:
import requests
import time

def check_endpoint(endpoint_url: str, endpoint_type: str, model_name: str, max_retries: int = 60) -> bool:
    """Check if endpoint is ready for requests."""
    test_payload = {
        "model": model_name,
        "prompt": "Hello" if endpoint_type == "completions" else None,
        "messages": [{"role": "user", "content": "Hello"}] if endpoint_type == "chat" else None,
        "max_tokens": 5
    }
    
    for _ in range(max_retries):
        try:
            response = requests.post(endpoint_url, json=test_payload, timeout=10)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    return False
```

## Module Structure Clarification

### Actual Package Structure:
```
packages/nemo-evaluator/src/nemo_evaluator/
├── __init__.py  # Exports: evaluate, show_available_tasks
├── api/
│   └── api_dataclasses.py
├── core/
│   ├── entrypoint.py  # Contains: show_available_tasks()
│   ├── evaluate.py
│   └── input.py  # Contains: get_available_evaluations() (internal)
```

### Documented (Incorrect) Structure:
```
packages/nemo-evaluator/src/nemo_eval/  # ❌ DOES NOT EXIST
└── utils/
    └── base.py  # ❌ DOES NOT EXIST
```

## Key Function Mappings

| Documented (Incorrect) | Actual Equivalent | Notes |
|------------------------|-------------------|-------|
| `nemo_eval.utils.base.list_available_evaluations()` | `nemo_evaluator.show_available_tasks()` | Returns `None`, prints to stdout |
| `nemo_eval.utils.base.find_framework()` | Use CLI: `nv-eval ls tasks` | No direct Python equivalent |
| `nemo_eval.utils.base.wait_for_fastapi_server()` | Manual implementation with `requests` | No built-in function |
| `nemo_eval.utils.base.check_endpoint()` | Manual implementation with `requests` | No built-in function |

## Impact Assessment

### High Priority ✅ COMPLETED
- [x] Main reference documentation (`evaluation-utils.md`)
- [x] Broken script (`scripts/helpers.py`)
- [x] All documentation cross-references

### Medium Priority ⚠️ PENDING
- [ ] Tutorial notebooks (require manual JSON editing)

### Low Priority ℹ️ INFORMATIONAL
- Documentation in `docs-archive/` (archived, less critical)

## Verification Steps

To verify corrections:

```bash
# 1. Check for remaining references to non-existent module
grep -r "nemo_eval.utils.base" docs/ --exclude-dir=docs-archive

# 2. Verify scripts can import correctly
python -c "from nemo_evaluator import show_available_tasks; print('✓ Import successful')"

# 3. Test helpers.py
python scripts/helpers.py  # Should not raise ImportError

# 4. Check tutorial notebooks
grep -r "nemo_eval.utils.base" tutorials/
```

## Recommendations for Future

1. **API Stability**: Consider stabilizing and documenting the public API in `nemo_evaluator`
2. **Version Migration Guide**: Create migration guide if `nemo_eval` was previously used
3. **Automated Tests**: Add tests that import from documented module paths to catch these issues
4. **Documentation Generation**: Consider auto-generating API docs from source code

## Related Issues

- Module naming inconsistency between `nemo_eval` and `nemo_evaluator`
- Lack of public utility functions for common operations (endpoint checking, server health)
- Tutorials depend on non-existent utility functions

---

**Audit Date**: October 1, 2025  
**Audit Methodology**: docs-audit-v3  
**Status**: Documentation corrections complete, tutorial notebooks require manual update


"""Built-in benchmarks -- all defined using the BYOB @benchmark + @scorer pattern.

These are first-party benchmarks that ship with nemo-evaluator. External users
use the exact same API to define their own benchmarks.
"""
# Import triggers @register() for each benchmark
import nemo_evaluator.benchmarks.drop  # noqa: F401
import nemo_evaluator.benchmarks.gpqa  # noqa: F401
import nemo_evaluator.benchmarks.gsm8k  # noqa: F401
import nemo_evaluator.benchmarks.healthbench  # noqa: F401
import nemo_evaluator.benchmarks.humaneval  # noqa: F401
import nemo_evaluator.benchmarks.math500  # noqa: F401
import nemo_evaluator.benchmarks.mgsm  # noqa: F401
import nemo_evaluator.benchmarks.mmlu  # noqa: F401
import nemo_evaluator.benchmarks.mmlu_pro  # noqa: F401
import nemo_evaluator.benchmarks.pinchbench  # noqa: F401
import nemo_evaluator.benchmarks.simpleqa  # noqa: F401
import nemo_evaluator.benchmarks.swebench_multilingual  # noqa: F401
import nemo_evaluator.benchmarks.swebench_verified  # noqa: F401
import nemo_evaluator.benchmarks.terminal_bench_v1  # noqa: F401
import nemo_evaluator.benchmarks.triviaqa  # noqa: F401
import nemo_evaluator.benchmarks.xstest  # noqa: F401

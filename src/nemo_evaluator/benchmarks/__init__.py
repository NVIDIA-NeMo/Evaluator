# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import nemo_evaluator.benchmarks.nmp_harbor  # noqa: F401
import nemo_evaluator.benchmarks.pinchbench  # noqa: F401
import nemo_evaluator.benchmarks.simpleqa  # noqa: F401
import nemo_evaluator.benchmarks.terminal_bench_v1  # noqa: F401
import nemo_evaluator.benchmarks.triviaqa  # noqa: F401
import nemo_evaluator.benchmarks.xstest  # noqa: F401

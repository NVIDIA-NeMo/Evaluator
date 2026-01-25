# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Fun messages for the wizard - because evaluations should be enjoyable!

To disable fun messages:
- Use --no-fun flag: nel wizard --no-fun
- Set environment variable: NEL_WIZARD_NO_FUN=1
"""

# Format: (emoji_version, text_version)
# Use {executor}, {deployment}, {count} placeholders as needed

EXECUTOR_MESSAGES = [
    (
        "🚀 Great choice! Let's configure the model next.",
        "[*] Great choice! Let's configure the model next.",
    ),
    (
        "⚡ {executor} it is! Now for the fun part.",
        "[!] {executor} it is! Now for the fun part.",
    ),
    (
        "💪 Ready to make those GPUs go brrr!",
        "[>] Ready to make those GPUs go brrr!",
    ),
    (
        "🎮 Game on! Let's set up the model.",
        "[>] Game on! Let's set up the model.",
    ),
    (
        "🏎️ {executor} selected! Buckle up!",
        "[>] {executor} selected! Buckle up!",
    ),
    (
        "⚙️ Nice! {executor} is a solid choice.",
        "[*] Nice! {executor} is a solid choice.",
    ),
    (
        "🔧 {executor} locked in! Time to pick a deployment.",
        "[>] {executor} locked in! Time to pick a deployment.",
    ),
    (
        "🎯 {executor} - excellent choice for this mission!",
        "[*] {executor} - excellent choice for this mission!",
    ),
]

DEPLOYMENT_MESSAGES = [
    (
        "🤖 Model deployment configured! Now pick your challenges.",
        "[*] Model deployment configured!",
    ),
    (
        "🧠 Brain ready! Time to choose the tests.",
        "[>] Brain ready! Time to choose the tests.",
    ),
    (
        "📡 Connection established! What should we evaluate?",
        "[*] Connection established!",
    ),
    (
        "🔌 {deployment} deployment ready! Let's pick some benchmarks.",
        "[>] {deployment} deployment ready!",
    ),
    (
        "🌐 Model endpoint configured! The stage is set.",
        "[*] Model endpoint configured!",
    ),
    (
        "🎬 Lights, camera, {deployment}! Time to select the scenes.",
        "[>] Lights, camera, {deployment}!",
    ),
]

TASKS_MESSAGES = [
    (
        "📚 {count} task(s) selected! Let's put that model to the test.",
        "[*] {count} task(s) selected!",
    ),
    (
        "🎯 Good luck to the model on these benchmarks!",
        "[>] Good luck to the model on these benchmarks!",
    ),
    (
        "🧪 Time to see what this model is made of!",
        "[!] Time to see what this model is made of!",
    ),
    (
        "🏋️ {count} challenges accepted! This will be interesting.",
        "[>] {count} challenges accepted!",
    ),
    (
        "📊 {count} benchmarks queued up. May the metrics be ever in your favor!",
        "[*] {count} benchmarks queued up.",
    ),
    (
        "🎲 The dice are cast! {count} task(s) ready to roll.",
        "[>] The dice are cast! {count} task(s) ready.",
    ),
    (
        "🔬 Science time! {count} experiment(s) configured.",
        "[*] Science time! {count} experiment(s) configured.",
    ),
    (
        "🎪 {count} task(s) in the arena! Let the evaluation begin!",
        "[>] {count} task(s) in the arena!",
    ),
    (
        "📝 {count} quiz(zes) ready! Hope the model studied!",
        "[*] {count} quiz(zes) ready!",
    ),
    (
        "🏆 {count} trial(s) awaiting! May the best weights win!",
        "[>] {count} trial(s) awaiting!",
    ),
]

SAVED_MESSAGES = [
    (
        "✨ Config saved! You're all set to run.",
        "[OK] Config saved! You're all set to run.",
    ),
    (
        "🎉 Looking good! Your evaluation is ready to roll.",
        "[OK] Looking good! Your evaluation is ready to roll.",
    ),
    (
        "🔥 Config locked and loaded!",
        "[OK] Config locked and loaded!",
    ),
    (
        "💾 Saved! Your config is safe and sound.",
        "[OK] Saved! Your config is safe and sound.",
    ),
    (
        "✅ All set! Time to let the model prove itself.",
        "[OK] All set! Time to let the model prove itself.",
    ),
    (
        "🎊 Configuration complete! Let the evaluation begin!",
        "[OK] Configuration complete!",
    ),
    (
        "📝 Written to disk! One small step for config, one giant leap for evaluation.",
        "[OK] Config written!",
    ),
    (
        "🚀 Config ready for liftoff! T-minus... whenever you're ready!",
        "[OK] Config ready for liftoff!",
    ),
    (
        "🎵 Config saved! *chef's kiss* Magnifico!",
        "[OK] Config saved! Magnifico!",
    ),
    (
        "🌟 Stellar config! The stars are aligned for evaluation.",
        "[OK] Stellar config!",
    ),
]

MESSAGES = {
    "executor": EXECUTOR_MESSAGES,
    "deployment": DEPLOYMENT_MESSAGES,
    "tasks": TASKS_MESSAGES,
    "saved": SAVED_MESSAGES,
}

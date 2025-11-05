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


# check if docker exists
command -v docker >/dev/null 2>&1 || { echo 'docker not found'; exit 1; }

# Initialize: remove killed jobs file from previous runs
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
killed_jobs_file="$script_dir/killed_jobs.txt"
rm -f "$killed_jobs_file"

{% for task in evaluation_tasks %}
# {{ task.job_id }} {{ task.name }}

task_dir="{{ task.output_dir }}"
artifacts_dir="$task_dir/artifacts"
logs_dir="$task_dir/logs"

mkdir -m 777 -p "$task_dir"
mkdir -m 777 -p "$artifacts_dir"
mkdir -m 777 -p "$logs_dir"

# Check if this job was killed
if [ -f "$killed_jobs_file" ] && grep -q "^{{ task.job_id }}$" "$killed_jobs_file"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Job {{ task.job_id }} ({{ task.name }}) was killed, skipping execution" | tee -a "$logs_dir/stdout.log"
else
    # Create pre-start stage file
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$logs_dir/stage.pre-start"

    # Debug contents of the eval factory command's config
    {{ task.eval_factory_command_debug_comment | indent(4) }}

    # Docker run with eval factory command
    (
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$logs_dir/stage.running"
        docker run --rm --shm-size=100g {{ extra_docker_args }} \
      --name {{ task.container_name }} \
      --volume "$artifacts_dir":/results \
      {% for env_var in task.env_vars -%}
      -e {{ env_var }} \
      {% endfor -%}
      {{ task.eval_image }} \
      bash -c '
        {{ task.eval_factory_command | indent(8) }} ;
        exit_code=$?
        chmod 777 -R /results;
        if [ "$exit_code" -ne 0 ]; then
            echo "The evaluation container failed with exit code $exit_code" >&2;
            exit "$exit_code";
        fi;
        echo "Container completed successfully" >&2;
        exit 0;
      ' > "$logs_dir/stdout.log" 2>&1
    exit_code=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $exit_code" > "$logs_dir/stage.exit"
) >> "$logs_dir/stdout.log" 2>&1


{% if auto_export_destinations %}
# Monitor job completion and auto-export
(
    # Give it a moment to ensure file is fully written
    sleep 1

    exit_code=$(tail -1 "$logs_dir/stage.exit" | cut -d' ' -f2)
    if [ "$exit_code" = "0" ]; then
        # Log auto-export activity to task logs
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Job {{ task.job_id }} completed successfully. Starting auto-export..." >> "$logs_dir/stdout.log"

        {% for dest in auto_export_destinations %}
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Exporting job {{ task.job_id }} to {{ dest }}..." >> "$logs_dir/stdout.log"
        nemo-evaluator-launcher export {{ task.job_id }} --dest {{ dest }} >> "$logs_dir/stdout.log" 2>&1
        if [ $? -eq 0 ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Export to {{ dest }} completed successfully" >> "$logs_dir/stdout.log"
        else
            echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Export to {{ dest }} failed" >> "$logs_dir/stdout.log"
        fi
        {% endfor %}

        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Auto-export completed for job {{ task.job_id }}" >> "$logs_dir/stdout.log"
    else
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Job {{ task.job_id }} failed with exit code $exit_code. Skipping auto-export." >> "$logs_dir/stdout.log"
    fi
)

{% endif %}
fi


{% endfor %}

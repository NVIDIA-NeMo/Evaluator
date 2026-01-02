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

# Create all directories and stdout.log files upfront before any container starts
{% for task in evaluation_tasks %}
task_dir="{{ task.output_dir }}"
artifacts_dir="$task_dir/artifacts"
logs_dir="$task_dir/logs"

mkdir -m 777 -p "$task_dir"
mkdir -m 777 -p "$artifacts_dir"
mkdir -m 777 -p "$logs_dir"
# Create stdout.log file upfront
touch "$logs_dir/client_stdout.log"
chmod 666 "$logs_dir/client_stdout.log"
{% endfor %}

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
        {% set has_sidecars = task.sidecars and (task.sidecars|length > 0) %}
        {% if has_sidecars and not task.deployment %}
        # Create a lightweight "netns anchor" container so sidecars + client can share localhost
        NETNS_ANCHOR_NAME="netns-{{ task.name }}-{{ task.job_id }}"
        docker run -d --rm --name "$NETNS_ANCHOR_NAME" alpine:3.19 sleep infinity > /dev/null
        NETNS_CONTAINER_NAME="$NETNS_ANCHOR_NAME"
        {% endif %}
        {% if task.deployment %}
        docker run --rm --shm-size=100g --gpus all {{ task.deployment.extra_docker_args }} \
        --name {{ task.deployment.container_name }} --entrypoint '' \
        -p {{ task.deployment.port }}:{{ task.deployment.port }} \
        {% for env_var in task.deployment.env_vars -%}
        -e {{ env_var }} \
        {% endfor -%}
        {% for mount in task.deployment.mounts -%}
        -v {{ mount }} \
        {% endfor -%}
        {{ task.deployment.image }} \
        {{ task.deployment.command }} > "$logs_dir/server_stdout.log" 2>&1 &

        SERVER_PID=$!
        SERVER_CONTAINER_NAME="{{ task.deployment.container_name }}"
        NETNS_CONTAINER_NAME="$SERVER_CONTAINER_NAME"

        date
        # wait for the server to initialize
        TIMEOUT=600
        ELAPSED=0
        while [[ "$(curl -s -o /dev/null -w "%{http_code}" {{ task.deployment.health_url }})" != "200" ]]; do
            kill -0 $SERVER_PID 2>/dev/null || { echo "Server process $SERVER_PID died"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 1" > "$logs_dir/stage.exit"; exit 1; }
            [ $ELAPSED -ge $TIMEOUT ] && { echo "Health check timeout after ${TIMEOUT}s"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 1" > "$logs_dir/stage.exit"; exit 1; }
            sleep 5
            ELAPSED=$((ELAPSED + 5))
        done
        date

        {% endif %}
        {% if has_sidecars %}
        # Start sidecars in the same network namespace as the server (or netns anchor)
        {% for sc in task.sidecars %}
        docker run --rm --shm-size=2g --network container:$NETNS_CONTAINER_NAME \
        --name {{ sc.container_name }} {{ extra_docker_args }} \
        {{ sc.image }} {{ sc.command }} > "$logs_dir/sidecar_{{ sc.name }}.log" 2>&1 &
        SIDECAR_{{ sc.name | upper }}_PID=$!

        # wait for the sidecar to initialize
        TIMEOUT=600
        ELAPSED=0
        while [[ "$(docker run --rm --network container:$NETNS_CONTAINER_NAME curlimages/curl:8.5.0 -s -o /dev/null -w "%{http_code}" {{ sc.health_url }})" != "200" ]]; do
            kill -0 $SIDECAR_{{ sc.name | upper }}_PID 2>/dev/null || { echo "Sidecar {{ sc.name }} process died"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 1" > "$logs_dir/stage.exit"; exit 1; }
            [ $ELAPSED -ge $TIMEOUT ] && { echo "Sidecar {{ sc.name }} health check timeout after ${TIMEOUT}s"; echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) 1" > "$logs_dir/stage.exit"; exit 1; }
            sleep 5
            ELAPSED=$((ELAPSED + 5))
        done
        {% endfor %}
        {% endif %}
        docker run --rm --shm-size=100g {{ extra_docker_args }} \
        {% if task.deployment or has_sidecars %}--network container:$NETNS_CONTAINER_NAME \{% endif %}--name {{ task.client_container_name }} \
      --volume "$artifacts_dir":/results \
      {% if task.dataset_mount_host and task.dataset_mount_container -%}
      --volume "{{ task.dataset_mount_host }}:{{ task.dataset_mount_container }}" \
      {% endif -%}
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
      ' > "$logs_dir/client_stdout.log" 2>&1
    exit_code=$?

    {% if has_sidecars %}
    # Stop sidecars
    {% for sc in task.sidecars %}
    docker stop {{ sc.container_name }} 2>/dev/null || true
    {% endfor %}
    {% endif %}

    {% if task.deployment %}
    # Stop the server
    docker stop $SERVER_CONTAINER_NAME 2>/dev/null || true
    {% endif %}

    {% if has_sidecars and not task.deployment %}
    # Stop netns anchor
    docker stop "$NETNS_ANCHOR_NAME" 2>/dev/null || true
    {% endif %}

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

# check if docker exists
command -v docker >/dev/null 2>&1 || { echo 'docker not found'; exit 1; }

set -eu

{% for task in evaluation_tasks %}

artifacts_dir={{ task.output_dir }}/artifacts
logs_dir={{ task.output_dir }}/logs

mkdir -m 777 -p "$artifacts_dir"
mkdir -p "$logs_dir"

# Create pre-start stage file
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$logs_dir/stage.pre-start"

# Start docker container in background with proper exit code capture
(
    set +e
    docker run --rm --shm-size=100g --name {{ task.container_name }} \
      --volume "$artifacts_dir":/results \
      {% for env_var in task.env_vars %}-e {{ env_var }} {% endfor %} \
      {{ task.eval_image }} \
      bash -c '
        {{ task.eval_factory_command }} ;
        exit_code=$?
        chmod 777 -R /results;
        if [ "$exit_code" -ne 0 ]; then
            echo "nv_eval failed with exit code $exit_code" >&2;
            exit "$exit_code";
        fi;
        echo "Container completed successfully" >&2;
        exit 0;
      ' > "$logs_dir/stdout.log" 2>&1
    exit_code=$?
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $exit_code" > "$logs_dir/stage.exit"
) >> "$logs_dir/stdout.log" 2>&1 &

# Create running stage file
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$logs_dir/stage.running"

{% if auto_export_destinations %}
# Monitor job completion and auto-export
(
    # Wait for the Docker container to finish by monitoring the stage.exit file
    while [ ! -f "$logs_dir/stage.exit" ]; do
        sleep 2
    done

    # Give it a moment to ensure file is fully written
    sleep 1

    exit_code=$(tail -1 "$logs_dir/stage.exit" | cut -d' ' -f2)
    if [ "$exit_code" = "0" ]; then
        # Log auto-export activity to task logs
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Job {{ task.job_id }} completed successfully. Starting auto-export..." >> "$logs_dir/stdout.log"

        {% for dest in auto_export_destinations %}
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Exporting job {{ task.job_id }} to {{ dest }}..." >> "$logs_dir/stdout.log"
        nv-eval export {{ task.job_id }} --dest {{ dest }} >> "$logs_dir/stdout.log" 2>&1
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
) &
{% endif %}
{% endfor %}

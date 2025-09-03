# Copyright (c) 2025, NVIDIA CORPORATION.
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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --package)
        PACKAGE="$2"
        shift 2
        ;;
    --python-version)
        PYTHON_VERSION="$2"
        shift 2
        ;;
    --use-uv)
        USE_UV="true"
        shift 1
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --package {nemo-evaluator} --python-version {3.10|3.11|3.12}"
        exit 1
        ;;
    esac
done

if [[ -z "${PACKAGE:-}" || -z "${PYTHON_VERSION:-}" ]]; then
    echo "Error: --package argument is required"
    echo "Usage: $0 --package {nemo-evaluator} --python-version {3.10|3.11|3.12}"
    exit 1
fi

if [[ "$PACKAGE" != "nemo-evaluator" || ("$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12") ]]; then
    echo "Error: --package must be either 'nemo-evaluator' and --python-version must be either '3.10', '3.11', or '3.12'"
    echo "Usage: $0 --package {nemo-evaluator} --python-version {3.10|3.11|3.12}"
    exit 1
fi

if [[ -z "${USE_UV:-}" ]]; then
    USE_UV="false"
fi


main() {
    if [[ -n "${PAT:-}" ]]; then
        echo -e "machine github.com\n  login token\n  password $PAT" >~/.netrc
        chmod 600 ~/.netrc
    fi

    # Install dependencies
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y curl

    # Install Python
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get install -y python$PYTHON_VERSION-dev python$PYTHON_VERSION-venv python3-pip
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_VERSION 1

    cd packages/$PACKAGE

    if [[ "$USE_UV" == "true" ]]; then
        # Install uv
        UV_VERSION="0.7.2"
        curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        export UV_PROJECT_ENVIRONMENT=/opt/venv
        export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
        export UV_LINK_MODE=copy

        # Create virtual environment and install dependencies
        uv venv ${UV_PROJECT_ENVIRONMENT} $([[ "$BASE_IMAGE" == "pytorch" ]] && echo "--system-site-packages")

        # Install dependencies
        uv sync --locked --only-group build ${UV_ARGS[@]}
        uv sync \
            --link-mode copy \
            --locked \
            --all-groups ${UV_ARGS[@]}

        # Install the package
        uv pip install --no-deps -e .
    else
        python -m venv /opt/venv
        /opt/venv/bin/pip install -e .
    fi


}

# Call the main function
main "$@"

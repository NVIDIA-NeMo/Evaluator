#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

main() {

    UV_ARGS=(
            "--no-install-package" "torch"
            "--no-install-package" "torchvision"
            "--no-install-package" "triton"
            "--no-install-package" "nvidia-cublas-cu12"
            "--no-install-package" "nvidia-cuda-cupti-cu12"
            "--no-install-package" "nvidia-cuda-nvrtc-cu12"
            "--no-install-package" "nvidia-cuda-runtime-cu12"
            "--no-install-package" "nvidia-cudnn-cu12"
            "--no-install-package" "nvidia-cufft-cu12"
            "--no-install-package" "nvidia-cufile-cu12"
            "--no-install-package" "nvidia-curand-cu12"
            "--no-install-package" "nvidia-cusolver-cu12"
            "--no-install-package" "nvidia-cusparse-cu12"
            "--no-install-package" "nvidia-cusparselt-cu12"
            "--no-install-package" "nvidia-nccl-cu12"
        )

    # Create virtual environment and install dependencies
    uv venv ${UV_PROJECT_ENVIRONMENT} --system-site-packages

    # Install dependencies
    uv sync --locked --only-group build ${UV_ARGS[@]}
    uv sync --locked --only-group te ${UV_ARGS[@]}
    uv sync \
        --link-mode copy \
        --all-groups ${UV_ARGS[@]}

    # Run install overrides
    bash docker/common/install_conflicting_deps.sh

    # Install the package
    uv pip install --no-deps -e .

    # Write environment variables to a file for later sourcing
    cat >/opt/venv/env.sh <<'EOF'
#!/bin/bash
export UV_PROJECT_ENVIRONMENT=/opt/venv
export PATH="/opt/venv/bin:$PATH"
export UV_LINK_MODE=copy
export PATH="/root/.local/bin:$PATH"
EOF

    chmod +x /opt/venv/env.sh
}

# Call the main function
main "$@"

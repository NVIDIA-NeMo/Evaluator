#!/usr/bin/env python3
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
"""Simple test script for partial Docker image pull functionality."""

import argparse
import json
import os
import sys

from nemo_evaluator_launcher.common.partial_pull import (
    GitlabRegistryAuthenticator,
    NvcrRegistryAuthenticator,
    RegistryAuthenticator,
    find_file_in_image_layers,
)


def parse_container_image(container_image: str) -> tuple[str, str, str]:
    """Parse a container image string into registry, repository, and tag.

    Args:
        container_image: Container image string (e.g., "nvcr.io/nvidia/eval-factory/lm-evaluation-harness:25.10")

    Returns:
        Tuple of (registry_url, repository, tag)
    """
    # Split tag from image
    if ":" in container_image:
        image_part, tag = container_image.rsplit(":", 1)
    else:
        image_part = container_image
        tag = "latest"

    # Parse registry and repository
    parts = image_part.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid container image format: {container_image}")

    # Check if first part is a registry (contains '.' or is 'localhost')
    if "." in parts[0] or parts[0] == "localhost":
        registry_host = parts[0]
        # Check if registry has a port
        if ":" in registry_host:
            registry_url = registry_host
        else:
            registry_url = registry_host
        repository = "/".join(parts[1:])
    else:
        # Default registry (Docker Hub)
        registry_url = "registry-1.docker.io"
        repository = image_part

    return registry_url, repository, tag


def create_authenticator(
    registry_type: str, registry_url: str, repository: str
) -> RegistryAuthenticator:
    """Create the appropriate authenticator based on registry type.

    Args:
        registry_type: Type of registry ('gitlab' or 'nvcr')
        registry_url: Registry URL
        repository: Repository name

    Returns:
        Authenticated registry authenticator instance
    """
    if registry_type == "gitlab":
        username = os.getenv("DOCKER_USERNAME")
        password = os.getenv("GITLAB_TOKEN")
        if not username or not password:
            print("❌ Please set DOCKER_USERNAME and GITLAB_TOKEN environment variables")
            print("Example:")
            print("  export DOCKER_USERNAME='gitlab-ci-token'")
            print("  export GITLAB_TOKEN='your_gitlab_token'")
            sys.exit(1)
        return GitlabRegistryAuthenticator(
            registry_url=registry_url,
            username=username,
            password=password,
            repository=repository,
        )
    elif registry_type == "nvcr":
        username = os.getenv("NVCR_USERNAME") or os.getenv("DOCKER_USERNAME")
        password = os.getenv("NVCR_PASSWORD") or os.getenv("NVCR_API_KEY")
        if not username or not password:
            print("❌ Please set NVCR_USERNAME and NVCR_PASSWORD (or NVCR_API_KEY) environment variables")
            print("Example:")
            print("  export NVCR_USERNAME='$oauthtoken'")
            print("  export NVCR_API_KEY='your_nvcr_api_key'")
            sys.exit(1)
        return NvcrRegistryAuthenticator(
            registry_url=registry_url,
            username=username,
            password=password,
        )
    else:
        raise ValueError(f"Unknown registry type: {registry_type}")


def main():
    """Test the partial pull functionality."""
    parser = argparse.ArgumentParser(
        description="Test partial Docker image pull functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GitLab registry (default)
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml

  # nvcr.io registry
  %(prog)s --registry nvcr --container nvcr.io/nvidia/eval-factory/simple-evals:25.10 --target-file /opt/metadata/framework.yml

  # Override max layer size (100MB)
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml --max-layer-size 104857600

  # Override tag
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml --tag custom-tag
        """,
    )
    parser.add_argument(
        "--registry",
        type=str,
        choices=["gitlab", "nvcr"],
        default="gitlab",
        help="Registry type: 'gitlab' or 'nvcr' (default: gitlab)",
    )
    parser.add_argument(
        "--harness",
        type=str,
        help="Harness name for GitLab registry (e.g., 'simple-evals', 'lm-evaluation-harness')",
    )
    parser.add_argument(
        "--container",
        type=str,
        help="Full container image (e.g., 'nvcr.io/nvidia/eval-factory/simple-evals:25.10'). "
        "Required for nvcr registry, optional for gitlab (overrides --harness).",
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default="/opt/metadata/framework.yml",
        help="Target file path to search for (default: /opt/metadata/framework.yml)",
    )
    parser.add_argument(
        "--max-layer-size",
        type=int,
        default=100 * 1024,  # 100KB
        help="Maximum layer size in bytes. Only layers smaller than this will be checked. (default: 100KB)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="dev-2025-11-10T10-21-8021b046",
        help="Image tag for GitLab registry (default: dev-2025-11-10T10-21-8021b046). "
        "Ignored if --container is provided.",
    )

    args = parser.parse_args()

    # Determine registry URL, repository, and tag
    if args.registry == "nvcr":
        if not args.container:
            print("❌ --container is required when using --registry nvcr")
            parser.print_help()
            sys.exit(1)
        try:
            registry_url, repository, tag = parse_container_image(args.container)
        except ValueError as e:
            print(f"❌ Error parsing container image: {e}")
            sys.exit(1)
    else:  # gitlab
        if args.container:
            # Parse container image
            try:
                registry_url, repository, tag = parse_container_image(args.container)
            except ValueError as e:
                print(f"❌ Error parsing container image: {e}")
                sys.exit(1)
        elif args.harness:
            # Use GitLab default format
            registry_url = "gitlab-master.nvidia.com:5005"
            repository = f"dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/{args.harness}"
            tag = args.tag
        else:
            print("❌ Please provide either --harness or --container for GitLab registry")
            parser.print_help()
            sys.exit(1)

    target_file = args.target_file
    max_layer_size = args.max_layer_size

    print("🔍 Docker Registry Partial Pull Test")
    print(f"  Registry type: {args.registry}")
    print(f"  Registry: {registry_url}")
    print(f"  Repository: {repository}")
    print(f"  Tag: {tag}")
    print(f"  Target file: {target_file}")
    print(
        f"  Max layer size: {max_layer_size} bytes ({max_layer_size / (1024 * 1024):.2f} MB)"
    )
    print()

    # Create authenticator
    auth = create_authenticator(args.registry, registry_url, repository)

    # Authenticate
    print("🔐 Authenticating...")
    if not auth.authenticate(repository=repository):
        print("❌ Authentication failed. Please check your credentials.")
        sys.exit(1)
    print("✅ Authentication successful")
    print()

    # Find the file in image layers
    print(f"🔍 Searching for {target_file} in image layers...")
    try:
        file_content = find_file_in_image_layers(
            authenticator=auth,
            repository=repository,
            reference=tag,
            target_file=target_file,
            max_layer_size=max_layer_size,
        )

        if file_content:
            print(f"✅ Found {target_file}!")
            print(f"📄 File content:")
            print(file_content)
            print()

            # Try to parse as JSON/YAML
            try:
                metadata = json.loads(file_content)
                print("📋 Parsed JSON:")
                print(json.dumps(metadata, indent=2))
            except json.JSONDecodeError:
                # Not JSON, might be YAML or plain text
                print("📋 File content (not JSON)")
        else:
            print(f"❌ File {target_file} not found in any layer")
            sys.exit(1)

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

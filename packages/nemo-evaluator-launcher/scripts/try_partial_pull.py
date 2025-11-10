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
    find_file_in_image_layers,
)


def main():
    """Test the partial pull functionality."""
    parser = argparse.ArgumentParser(
        description="Test partial Docker image pull functionality for GitLab registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for a file in a harness container (default: 100KB max layer size)
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml

  # Override max layer size (100MB)
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml --max-layer-size 104857600

  # Override tag
  %(prog)s --harness simple-evals --target-file /opt/metadata/framework.yml --tag custom-tag
        """,
    )
    parser.add_argument(
        "--harness",
        type=str,
        required=True,
        help="Harness name (e.g., 'simple-evals', 'lm-evaluation-harness')",
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
        help="Image tag (default: dev-2025-11-10T10-21-8021b046)",
    )

    args = parser.parse_args()

    # Get credentials from environment
    username = os.getenv("DOCKER_USERNAME")
    password = os.getenv("GITLAB_TOKEN")

    if not username or not password:
        print("❌ Please set DOCKER_USERNAME and GITLAB_TOKEN environment variables")
        print("Example:")
        print("  export DOCKER_USERNAME='gitlab-ci-token'")
        print("  export GITLAB_TOKEN='your_gitlab_token'")
        sys.exit(1)

    # GitLab registry configuration
    registry_url = "gitlab-master.nvidia.com:5005"
    repository = f"dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/{args.harness}"
    tag = args.tag
    target_file = args.target_file
    max_layer_size = args.max_layer_size

    print("🔍 Docker Registry Partial Pull Test")
    print(f"  Registry: {registry_url}")
    print(f"  Repository: {repository}")
    print(f"  Tag: {tag}")
    print(f"  Target file: {target_file}")
    print(f"  Username: {username}")
    print(f"  Password set: {'Yes' if password else 'No'}")
    print(
        f"  Max layer size: {max_layer_size} bytes ({max_layer_size / (1024 * 1024):.2f} MB)"
    )
    print()

    # Initialize authenticator
    auth = GitlabRegistryAuthenticator(
        registry_url=registry_url,
        username=username,
        password=password,
        repository=repository,
    )

    # Authenticate
    print("🔐 Authenticating...")
    if not auth.authenticate():
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

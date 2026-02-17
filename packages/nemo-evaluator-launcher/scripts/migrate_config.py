#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Thin shim — the real implementation lives in the package.
# Prefer using the installed entrypoint: nel-migrate-config
#
# TODO(Apr 2026): Remove this shim once the migration period is over.

from nemo_evaluator_launcher.cli.migrate_config import main

if __name__ == "__main__":
    main()

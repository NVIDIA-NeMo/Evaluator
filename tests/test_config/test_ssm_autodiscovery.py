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
"""Tests for ECS config auto-discovery via SSM Parameter Store."""

import json
from unittest.mock import MagicMock, patch

import pytest

from nemo_evaluator.config import EcsFargateSandbox, SshSidecarConfig

# ── Fixtures ──────────────────────────────────────────────────────────

SSM_CONFIG = {
    "cluster": "harbor-cluster",
    "subnets": ["subnet-aaa", "subnet-bbb"],
    "security_groups": ["sg-123"],
    "assign_public_ip": True,
    "execution_role_arn": "arn:aws:iam::123456:role/exec",
    "task_role_arn": "arn:aws:iam::123456:role/task",
    "log_group": "/ecs/harbor/us-west-2",
    "s3_bucket": "harbor-staging-bucket",
    "ecr_repository": "123456.dkr.ecr.us-west-2.amazonaws.com/harbor",
    "codebuild_service_role": "arn:aws:iam::123456:role/codebuild",
    "dockerhub_secret_arn": "arn:aws:secretsmanager:us-west-2:123456:secret:dh",
    "efs_filesystem_id": "fs-aaa",
    "efs_access_point_id": "fsap-bbb",
    "ssh_sidecar": {
        "sshd_port": 2222,
        "exec_server_port": 5000,
        "public_key_secret_arn": "arn:aws:secretsmanager:us-west-2:123456:secret:pub",
        "private_key_secret_arn": "arn:aws:secretsmanager:us-west-2:123456:secret:priv",
    },
}


def _mock_ssm_client(param_value: dict):
    """Create a mock boto3 SSM client that returns param_value as JSON."""
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.return_value = {"Parameter": {"Value": json.dumps(param_value)}}
    return mock_ssm


def _mock_boto3(ssm_client):
    """Create a mock boto3 module."""
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = ssm_client
    return mock_boto3


@pytest.fixture(autouse=True)
def _clear_ssm_cache():
    """Clear the SSM config cache before each test."""
    from nemo_evaluator.sandbox.ecs_fargate import _ssm_config_cache

    _ssm_config_cache.clear()
    yield
    _ssm_config_cache.clear()


# ── resolve_ecs_config_from_ssm tests ─────────────────────────────────


class TestResolveEcsConfigFromSsm:
    def test_basic_resolution(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm_client = _mock_ssm_client(SSM_CONFIG)
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), type("CE", (Exception,), {}))
            result = resolve_ecs_config_from_ssm("us-west-2", "harbor")

        assert result["cluster"] == "harbor-cluster"
        assert result["subnets"] == ["subnet-aaa", "subnet-bbb"]
        ssm_client.get_parameter.assert_called_once_with(Name="/harbor/ecs-sandbox/config")

    def test_custom_project_name(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm_client = _mock_ssm_client(SSM_CONFIG)
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), type("CE", (Exception,), {}))
            resolve_ecs_config_from_ssm("eu-west-1", "myproject")

        ssm_client.get_parameter.assert_called_once_with(Name="/myproject/ecs-sandbox/config")

    def test_caching(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm_client = _mock_ssm_client(SSM_CONFIG)
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), type("CE", (Exception,), {}))
            r1 = resolve_ecs_config_from_ssm("us-west-2")
            r2 = resolve_ecs_config_from_ssm("us-west-2")

        assert r1 is r2
        assert ssm_client.get_parameter.call_count == 1

    def test_different_regions_not_cached_together(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm_client = _mock_ssm_client(SSM_CONFIG)
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), type("CE", (Exception,), {}))
            resolve_ecs_config_from_ssm("us-west-2")
            resolve_ecs_config_from_ssm("us-east-1")

        assert ssm_client.get_parameter.call_count == 2

    def test_parameter_not_found_raises(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ClientError = type(
            "ClientError",
            (Exception,),
            {
                "__init__": lambda self, **kw: (
                    setattr(self, "response", kw.get("response", {})) or Exception.__init__(self, "not found")
                ),
            },
        )
        ssm_client = MagicMock()
        err = ClientError(response={"Error": {"Code": "ParameterNotFound"}})
        ssm_client.get_parameter.side_effect = err
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), ClientError)
            with pytest.raises(RuntimeError, match="SSM parameter.*not found"):
                resolve_ecs_config_from_ssm("us-west-2")

    def test_invalid_json_raises(self):
        from nemo_evaluator.sandbox.ecs_fargate import resolve_ecs_config_from_ssm

        ssm_client = MagicMock()
        ssm_client.get_parameter.return_value = {"Parameter": {"Value": "not-json{"}}
        mock_boto = _mock_boto3(ssm_client)

        with patch("nemo_evaluator.sandbox.ecs_fargate._require_aws_sdks") as mock_aws:
            mock_aws.return_value = (mock_boto, MagicMock(), type("CE", (Exception,), {}))
            with pytest.raises(RuntimeError, match="invalid JSON"):
                resolve_ecs_config_from_ssm("us-west-2")


# ── Config model tests ────────────────────────────────────────────────


class TestEcsFargateSandboxModel:
    def test_cluster_optional(self):
        cfg = EcsFargateSandbox(region="us-west-2")
        assert cfg.cluster is None
        assert cfg.region == "us-west-2"

    def test_cluster_explicit(self):
        cfg = EcsFargateSandbox(cluster="my-cluster")
        assert cfg.cluster == "my-cluster"

    def test_ssm_project_default(self):
        cfg = EcsFargateSandbox(region="us-west-2")
        assert cfg.ssm_project == "harbor"

    def test_ssm_project_custom(self):
        cfg = EcsFargateSandbox(region="us-west-2", ssm_project="myproject")
        assert cfg.ssm_project == "myproject"

    def test_assign_public_ip_none_by_default(self):
        cfg = EcsFargateSandbox(region="us-west-2")
        assert cfg.assign_public_ip is None

    def test_max_task_lifetime_none_by_default(self):
        cfg = EcsFargateSandbox(region="us-west-2")
        assert cfg.max_task_lifetime_sec is None


# ── SSM merge logic tests (via _build_ecs_sandbox_config) ─────────────


class TestBuildEcsSandboxConfigWithSsm:
    def _build(self, cfg: EcsFargateSandbox):
        from nemo_evaluator.orchestration.orchestrator import _build_ecs_sandbox_config

        return _build_ecs_sandbox_config(cfg)

    def test_full_autodiscovery(self):
        """region only → everything from SSM."""
        cfg = EcsFargateSandbox(region="us-west-2")

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.cluster == "harbor-cluster"
        assert result.subnets == ["subnet-aaa", "subnet-bbb"]
        assert result.security_groups == ["sg-123"]
        assert result.assign_public_ip is True
        assert result.execution_role_arn == "arn:aws:iam::123456:role/exec"
        assert result.task_role_arn == "arn:aws:iam::123456:role/task"
        assert result.log_group == "/ecs/harbor/us-west-2"
        assert result.s3_bucket == "harbor-staging-bucket"
        assert result.ecr_repository == "123456.dkr.ecr.us-west-2.amazonaws.com/harbor"
        assert result.codebuild_service_role == "arn:aws:iam::123456:role/codebuild"
        assert result.dockerhub_secret_arn == "arn:aws:secretsmanager:us-west-2:123456:secret:dh"
        assert result.efs_filesystem_id == "fs-aaa"
        assert result.efs_access_point_id == "fsap-bbb"
        assert result.max_task_lifetime_sec == 14400

    def test_ssh_sidecar_autodiscovery(self):
        """SSH sidecar key ARNs and exec_server_port populated from SSM when not in YAML."""
        cfg = EcsFargateSandbox(region="us-west-2")

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.ssh_sidecar is not None
        assert result.ssh_sidecar.sshd_port == 2222
        assert result.ssh_sidecar.exec_server_port == 5000
        assert "pub" in result.ssh_sidecar.public_key_secret_arn
        assert "priv" in result.ssh_sidecar.private_key_secret_arn

    def test_ssh_sidecar_exec_server_port_defaults_to_5000(self):
        """When SSM omits exec_server_port, it defaults to 5000 (exec-server mode)."""
        ssm_no_exec_port = dict(SSM_CONFIG)
        ssm_no_exec_port["ssh_sidecar"] = {
            "sshd_port": 2222,
            "public_key_secret_arn": SSM_CONFIG["ssh_sidecar"]["public_key_secret_arn"],
            "private_key_secret_arn": SSM_CONFIG["ssh_sidecar"]["private_key_secret_arn"],
        }
        cfg = EcsFargateSandbox(region="us-west-2")

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=ssm_no_exec_port,
        ):
            result = self._build(cfg)

        assert result.ssh_sidecar is not None
        assert result.ssh_sidecar.exec_server_port == 5000

    def test_yaml_overrides_ssm(self):
        """Explicitly set YAML fields take precedence over SSM."""
        cfg = EcsFargateSandbox(
            region="us-west-2",
            ecr_repository="my-custom-ecr-repo",
            log_group="/my/custom/log-group",
            assign_public_ip=False,
        )

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.ecr_repository == "my-custom-ecr-repo"
        assert result.log_group == "/my/custom/log-group"
        assert result.assign_public_ip is False
        assert result.cluster == "harbor-cluster"

    def test_yaml_subnets_override_ssm(self):
        """Non-empty subnets list in YAML overrides SSM."""
        cfg = EcsFargateSandbox(
            region="us-west-2",
            subnets=["subnet-override"],
        )

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.subnets == ["subnet-override"]

    def test_ssh_sidecar_yaml_overrides_ssm_keys(self):
        """User-specified ssh_sidecar key ARNs override SSM."""
        cfg = EcsFargateSandbox(
            region="us-west-2",
            ssh_sidecar=SshSidecarConfig(
                public_key_secret_arn="arn:custom:pub",
                private_key_secret_arn="arn:custom:priv",
            ),
        )

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.ssh_sidecar.public_key_secret_arn == "arn:custom:pub"
        assert result.ssh_sidecar.private_key_secret_arn == "arn:custom:priv"

    def test_ssh_sidecar_yaml_partial_inherits_ssm(self):
        """ssh_sidecar in YAML with only one ARN → other filled from SSM."""
        cfg = EcsFargateSandbox(
            region="us-west-2",
            ssh_sidecar=SshSidecarConfig(
                public_key_secret_arn="arn:custom:pub",
                private_key_secret_arn="",
            ),
        )

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ):
            result = self._build(cfg)

        assert result.ssh_sidecar.public_key_secret_arn == "arn:custom:pub"
        assert result.ssh_sidecar.private_key_secret_arn == SSM_CONFIG["ssh_sidecar"]["private_key_secret_arn"]

    def test_no_ssm_when_cluster_specified(self):
        """Explicit cluster → SSM is NOT consulted."""
        cfg = EcsFargateSandbox(
            region="us-west-2",
            cluster="explicit-cluster",
            assign_public_ip=True,
            max_task_lifetime_sec=7200,
        )

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
        ) as mock_ssm:
            result = self._build(cfg)

        mock_ssm.assert_not_called()
        assert result.cluster == "explicit-cluster"
        assert result.assign_public_ip is True
        assert result.max_task_lifetime_sec == 7200

    def test_no_ssm_when_no_region(self):
        """No region → SSM is NOT consulted, cluster defaults to empty."""
        cfg = EcsFargateSandbox(cluster="my-cluster")

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
        ) as mock_ssm:
            result = self._build(cfg)

        mock_ssm.assert_not_called()
        assert result.cluster == "my-cluster"

    def test_defaults_when_no_ssm(self):
        """Without SSM, None fields get sensible defaults."""
        cfg = EcsFargateSandbox(cluster="c")

        result = self._build(cfg)

        assert result.assign_public_ip is True
        assert result.max_task_lifetime_sec == 14400
        assert result.cluster == "c"

    def test_custom_ssm_project(self):
        """ssm_project changes the SSM parameter path prefix."""
        cfg = EcsFargateSandbox(region="us-west-2", ssm_project="myproject")

        with patch(
            "nemo_evaluator.sandbox.ecs_fargate.resolve_ecs_config_from_ssm",
            return_value=SSM_CONFIG,
        ) as mock_ssm:
            self._build(cfg)

        mock_ssm.assert_called_once_with("us-west-2", "myproject")

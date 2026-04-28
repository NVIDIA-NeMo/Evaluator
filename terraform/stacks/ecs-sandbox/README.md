# Stack: `ecs-sandbox`

Root stack for the AWS infrastructure consumed by NEL's `ECSFargateSandbox` backend (see [Sandbox Orchestration → ECSFargateSandbox](../../../docs/architecture/sandbox.md#ecsfargatesandbox)). Creates global IAM (shared across regions) and instantiates [`modules/ecs-sandbox-region`](../../modules/ecs-sandbox-region) once per region listed in `var.regions`. NEL itself is **not** deployed here.

## Files

| File | Purpose |
|------|---------|
| [`provider.tf`](provider.tf) | Default AWS provider (uses `var.regions[0]`) plus 11 region-specific aliases. |
| [`main.tf`](main.tf) | Stable `region -> CIDR index` map, shared TLS keypair (ED25519), and one `module` block per region. Modules are gated by `count = contains(var.regions, "...") ? 1 : 0`. |
| [`iam.tf`](iam.tf) | ECS execution role, ECS task role, CodeBuild service role, orchestrator IAM user + access key + custom policy. |
| [`outputs.tf`](outputs.tf) | Shared IAM outputs and per-region maps (VPC ID, SSM config parameter name). |
| [`variables.tf`](variables.tf) | All knobs with sane defaults. |
| [`versions.tf`](versions.tf) | Terraform >= 1.5, `aws ~> 5.0`, `tls ~> 4.0`. |
| [`backend.tf.example`](backend.tf.example) | Optional remote-state stub (commented). |

## Usage

```sh
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars

terraform init
terraform plan -out tfplan
terraform apply tfplan
```

A single-region apply takes roughly **3–5 minutes** (most of it is EFS mount-target propagation when `enable_efs = true`). Eleven regions in parallel is around **8–10 minutes**.

## Sample minimal `terraform.tfvars`

```hcl
project = "my-org-eval"
regions = ["us-west-2"]
enable_efs = false

tags = {
  Project   = "my-org-eval"
  ManagedBy = "terraform"
  Owner     = "platform-team"
}
```

## After apply: configuring NEL

Each region exposes a JSON config blob in SSM at `/{project}/ecs-sandbox/config`. NEL reads it directly — you do **not** copy values into a YAML config. To verify:

```sh
aws ssm get-parameter \
  --name "/$(terraform output -raw -json | jq -r '.regions.value[0] | "\(.)"')/ecs-sandbox/config" \
  --region us-west-2 \
  --query Parameter.Value \
  --output text | jq
```

The orchestrator user's credentials are emitted as outputs (`orchestrator_access_key_id` / `orchestrator_secret_access_key`). Wire them into your CI secrets store or `~/.aws/credentials` under a named profile.

## Adding/removing regions

Append to or remove from `var.regions`. Because each module is gated by `contains(var.regions, "...") ? 1 : 0`, Terraform creates only what's listed and destroys what's removed. The CIDR index map in `main.tf` is **stable**: each region always gets the same `/16` regardless of which other regions are active. To support a region not in the map, add it both to `provider.tf`, the `region_cidr_index` local in `main.tf`, and a new `module` block.

## Cleanup

```sh
terraform destroy
```

`force_destroy` on ECR and S3 plus `recovery_window_in_days = 0` on Secrets Manager mean nothing lingers after destroy. The orchestrator IAM access key is also removed.

## Pitfalls

- **First-time `init` takes minutes** because Terraform downloads the AWS provider binary for each aliased provider configuration even though the providers share a binary. This is expected.
- **Default provider region** comes from `var.regions[0]`. If you reorder the list, you reorder the home for IAM-tagged audit events. IAM resources themselves are global so the actual ARNs don't change.
- **`enable_efs` toggling is destructive** — flipping it off replaces the workspace filesystem.

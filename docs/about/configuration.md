(configuration)=

# Configuration

NeMo Evaluator stores persistent settings in a YAML config file.

## Config File Location

The config file is located at `~/.config/nemo-evaluator/config.yaml`.

If the `XDG_CONFIG_HOME` environment variable is set, the file is located at `$XDG_CONFIG_HOME/nemo-evaluator/config.yaml` instead.

## Managing Configuration

Use the `nemo-evaluator-launcher config` subcommand to manage settings:

```bash
nemo-evaluator-launcher config set <key> <value>   # set a value
nemo-evaluator-launcher config get <key>            # show effective value
nemo-evaluator-launcher config show                 # dump full config file
```

## Available Settings

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Key
  - Default
  - Description
* - `telemetry.level`
  - `2`
  - Telemetry reporting level (0=off, 1=minimal, 2=full). See {ref}`telemetry`.
```

## Example

```yaml
telemetry:
  level: 1
```

# Built-in config fragments

These YAML fragments ship with `nemo-evaluator` and can be referenced from the
`defaults:` list in any eval config:

```yaml
defaults:
  - clusters/local     # uses this built-in fragment
  - _self_
```

## Adding custom fragments

Place your own fragments in any of these locations (searched in order):

1. `conf/` directory next to your config file
2. `~/.config/nemo-evaluator/conf/`
3. These built-in defaults (lowest priority)

The directory name maps to the top-level config key:

| Directory     | Config key   |
|---------------|--------------|
| `services/`   | `services:`  |
| `clusters/`   | `cluster:`   |
| `sandboxes/`  | `sandboxes:` |
| `output/`     | `output:`    |

See `docs/tutorials/config-composition.md` for the full guide.

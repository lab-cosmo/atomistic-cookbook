# atomistic-cookbook-utils

Small shared utilities used by the recipes in the
[Atomistic Cookbook](https://atomistic-cookbook.org).

The goal is to keep recipe code focused on the science: any boilerplate
that would otherwise be copy-pasted across many recipes lives here.

## Installation

```bash
pip install atomistic-cookbook-utils
```

For a recipe's `environment.yml`, add it under `pip:` with a semantic
version range, e.g.

```yaml
- atomistic-cookbook-utils >=0.1,<0.2
```

## Contents

### `download_with_retry`

Download a file with automatic retries on transient HTTP errors
(429, 500, 502, 503, 504). Skips the download if the file already
exists, creates parent directories as needed.

```python
from atomistic_cookbook_utils import download_with_retry

download_with_retry(
    "https://zenodo.org/records/10566825/files/gaas_training.xyz",
    "data/gaas_training.xyz",
)
```

### `run_command`

Run a shell-like command string without invoking a shell. The string is
split with `shlex.split`, so quoting works as expected; pipes, redirection
and globs do *not* — for those, use `subprocess` directly.

```python
from atomistic_cookbook_utils import run_command

run_command("mtt train options.yaml")
run_command("ls -la", cwd="data", capture_output=True)
```

## Source

This package is developed inside the cookbook repository at
`src/atomistic-cookbook-utils/`. See `CONTRIBUTING.rst` at the repo
root for development workflow.

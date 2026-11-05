# Development

This page is for working on geo-trax itself: installing from a source checkout, running the tests and the linter, and building this documentation site. To use geo-trax, the PyPI install on the [home page](index.md#install) is enough.

## Install from source

Geo-trax needs Python 3.11 to 3.13. Clone the repository (add `--depth 1` for the latest snapshot only), create an environment, and install in editable mode (`-e`), so code changes take effect without reinstalling:

```bash
git clone https://github.com/rfonod/geo-trax.git
cd geo-trax
```

=== "pip"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate           # Windows: .venv\Scripts\activate
    python -m pip install -e '.[dev]'
    ```

=== "uv"

    ```bash
    uv venv --python 3.11
    source .venv/bin/activate           # Windows: .venv\Scripts\activate
    uv pip install -e '.[dev]'
    ```

=== "conda"

    ```bash
    conda create -n geo-trax python=3.11 -y
    conda activate geo-trax
    python -m pip install -e '.[dev]'
    ```

=== "Poetry"

    ```bash
    poetry install --extras dev         # creates and manages its own virtualenv
    ```

Use `python -m pip` rather than a bare `pip`: a `pip` found first on the `PATH` can belong to another interpreter (a conda base, say) even with the environment activated, and then installs geo-trax there instead.

The optional dependency groups combine freely, e.g. `python -m pip install -e '.[dev,docs]'`:

| Extra | Adds | Needed for |
|---|---|---|
| `dev` | pytest, ruff, and the Ultralytics development tools | running the tests and the linter |
| `docs` | MkDocs, Material for MkDocs, mkdocstrings, mkdocs-gen-files | building this site; needs a source checkout |
| `export` | the Ultralytics ONNX export backend | `train/export.sh` |
| `sahi` | SAHI | `--sahi` sliced inference |

The default detector downloads from [Hugging Face](https://huggingface.co/rfonod/geo-trax) on first use, into `~/.cache/huggingface/hub` unless `HF_HOME` or `HF_HUB_CACHE` says otherwise. CUDA-accelerated image matching needs a source-built OpenCV that the regular install would overwrite; see [GPU acceleration](guides/gpu.md) before installing into such an environment.

## Tests

```bash
pytest                                  # the whole suite in tests/
pytest tests/test_extract.py            # one module
pytest -k class_conf                    # tests whose name matches an expression
pytest -x -q                            # stop at the first failure, quieter output
```

The suite is made of fast unit tests: it runs neither the detector nor the full pipeline, so it needs no GPU and no model download. A fixture sets `GEOTRAX_DISABLE_UPDATE_CHECK` and `STABILO_DISABLE_UPDATE_CHECK`, so no test reaches PyPI. CI runs it on Ubuntu (Python 3.11 and 3.13), macOS and Windows (Python 3.11).

For an end-to-end check, run the pipeline on the 5-second sample clip that ships with the repository; it takes a few minutes on a laptop CPU and writes to `data/results/`, which git ignores:

```bash
geotrax batch data/U_video_cut.mp4 --no-geo
```

[`data/README.md`](https://github.com/rfonod/geo-trax/blob/main/data/README.md) explains how to download the orthophotos, master frames and segmentations for the full pipeline.

## Linting and formatting

```bash
ruff check .                            # lint
ruff format .                           # format
```

Both read their settings from `pyproject.toml`: a line length of 120, existing quote styles preserved, and the `E`, `F`, `W`, `A`, `B`, `PLC`, `PLE`, `PLW`, `I` and `UP` rule sets. CI currently enforces only the bug-level rules (`ruff check --select F,E9 .`), so a change must at least pass that.

## Documentation site

The site is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from `docs/` and `mkdocs.yml`:

```bash
python -m pip install -e '.[docs]'
mkdocs serve                            # live preview; prints the local address
mkdocs build --strict                   # the check CI runs; must finish with zero warnings
```

The CLI reference imports every pipeline stage, so the docs build installs the full package. Installing the CPU-only PyTorch wheels first, as CI does, keeps that environment small: `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

Several pages are generated on every build by `docs/_scripts/gen_reference.py` and must not be edited by hand; change their source instead:

| Pages | Generated from |
|---|---|
| [Config reference](reference/config/index.md) | the inline comments of `geotrax/cfg/default.yaml` |
| Preset pages | the differences between each preset and `default.yaml` |
| [CLI reference](reference/cli/index.md) | each stage module's `_build_parser()` |
| [Citation](citation.md) | `CITATION.cff` |

The build also checks that code and docs agree, and `--strict` turns each mismatch into a failure: a config key without an inline comment, a `— CLI: --flag` hint in a comment that names a flag no parser maps to that key (or a config-backed flag whose key has no such hint), README BibTeX that no longer matches `CITATION.cff`, and a public function of a module with an [API reference](reference/api/index.md) page that the page does not list. Every page must also appear in the `nav` of `mkdocs.yml`.

The API reference renders docstrings with [mkdocstrings](https://mkdocstrings.github.io/), so public helpers in those modules use Google-style docstrings (`Args:`, `Returns:`, `Raises:`, `Examples:`).

## Changing the configuration

The four bundled configs in `geotrax/cfg/` are complete, standalone files. `confident.yaml`, `lenient.yaml` and `stable.yaml` are identical to `default.yaml`, comments included, except for their header and their tuned values, so a change to `default.yaml` belongs in all four. The tests check that every config-backed CLI flag maps to a key of `default.yaml` and that each preset loads without needing defaults filled in, and the docs build checks the comments as described above.

A new config-backed flag is registered with `add_cfg_arg()` in `geotrax/utils/cli_utils.py`, which also writes the "Defaults to cfg -> ..." sentence of its help text; add the matching `— CLI: --flag` hint to the key's comment in all four configs.

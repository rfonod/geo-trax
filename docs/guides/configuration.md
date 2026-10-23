# Configuration

Every `geotrax` command reads one self-contained YAML file that holds all pipeline settings: detection, tracking, stabilization, georeferencing, visualization and plotting. The [config reference](../reference/config/index.md) lists every key with its default and meaning; this page covers how to pick a config, change it, and override it for a single run.

## Presets

Four configs ship with the package. Select one by name with `-c`:

| Preset | Focus |
|---|---|
| `default` | Balanced baseline, used when `-c` is not given |
| [`confident`](../reference/config/presets/confident.md) | Precision: fewer, more reliable detections |
| [`lenient`](../reference/config/presets/lenient.md) | Recall: catches more vehicles, with more false positives |
| [`stable`](../reference/config/presets/stable.md) | Stabilization quality, at a speed cost |

Each preset page lists the keys it changes. `-c` takes a preset name (the `.yaml` suffix is optional) or a path to your own file:

```bash
geotrax batch video.mp4 -c confident         # a bundled preset
geotrax batch video.mp4 -c ./my_config.yaml  # your own file
```

## Your own config

`geotrax config` locates, prints and copies the bundled files:

```bash
geotrax config show              # where the bundled configs live, and the preset names
geotrax config show lenient      # print one preset in full, with its comments
geotrax config copy              # copy every preset here as <name>_copy.yaml
geotrax config copy -o ~/myproj  # copy into another folder instead
```

Edit the copy and pass it with `-c`. A config only needs the keys you change: every key it leaves out takes its value from the bundled `default.yaml` (not from the preset the copy started as), and the log file lists the keys filled in this way. So a small file such as

```yaml
georef:
  transformation:
    target_crs: epsg:32633
ultralytics:
  conf: 0.35
```

is a complete config, and a copy made before an upgrade keeps working, with any keys added since at their defaults. A key that a newer Ultralytics release has removed, such as `ultralytics.keras` in copies made before v1.6.0, is ignored with a warning that asks you to delete it.

## One-off overrides with `--set`

`--set KEY=VALUE` changes one config value for a single run, without copying a file. It is available on every command that takes `-c`, and can be repeated:

```bash
geotrax extract video.mp4 --set conf=0.35 --set iou=0.6
geotrax extract video.mp4 --set tracker.botsort.track_buffer=45
geotrax georeference video.mp4 -orf ortho/ --set matching.downsample_ratio=0.25
geotrax batch PROCESSED/ --set stabilo.max_features=4000 --set visualization.tail_length=90
```

- **KEY** is a dotted path (`ultralytics.conf`) or any unambiguous tail of one (`conf`, `matching.gpu`). A misspelled key stops the run with suggestions, and an ambiguous one lists every candidate: `track_buffer` alone matches all six trackers, so write `tracker.botsort.track_buffer`. The full dotted path is the stable form, since a short name that is unique today can become ambiguous when a later release adds a key.
- **VALUE** is read with YAML rules, exactly as in the config file, so `true`, `null`, `0.35`, `[0, 1, 2]` and `{1: 0.15}` all mean what they look like. Quote values with spaces or braces for your shell: `--set 'class_conf={1: 0.15, 2: 0.4}'`.
- **Types** are checked against the current value: text where a number belongs, or a scalar where a list belongs, is rejected. Integers and decimals are interchangeable, `0`/`1` are accepted for booleans, and `key=` sets a text value to the empty string. `null` is accepted for the keys that may be null: those shipped as `null`, and the few whose description says so although they ship with a value, such as `cutout_width_px`.

Every applied override is printed, so none is silent.

### Flags and `--set`

Many keys also have a dedicated flag, such as `--conf` for `ultralytics.conf`; the [CLI reference](../reference/cli/index.md) maps each flag to its key. Prefer the flag where one exists: it validates its input (`--stab-detector xfeatt` fails at once and lists the valid detectors, while `--set` can only check the type) and it appears in `--help`. Precedence is simply CLI over config file. Passing a flag and `--set` for the same key is an error rather than a precedence puzzle:

```console
$ geotrax extract video.mp4 --set conf=0.1 --conf 0.5
CRITICAL - Conflicting overrides for 'ultralytics.conf':
  --set conf=0.1
  --conf 0.5
Pass only one.
```

`--set` is meant for experiments; to make a change durable, put it in a copied config.

## What a run records

`geotrax extract` writes `<video>.yaml` to the output folder, next to the tracks. It records the run (geo-trax version, time, source video, config file, output folder), the number of vehicles and track rows saved, the model as configured and as resolved, the class names, and every setting as the run actually used it, after all flags and `--set` overrides. It is a record for provenance; it is laid out differently from a config, so it cannot be passed back with `-c`.

## Logs

The console shows warnings, errors and a few key messages, such as applied overrides and model downloads; `--verbose` (`-v`) adds progress details. Every command also writes a log file that always includes those details, to a platform-specific folder:

| Platform | Default log folder |
|---|---|
| macOS | `~/Library/Logs/geo-trax` |
| Linux | `$XDG_STATE_HOME/geo-trax/logs` (default `~/.local/state/geo-trax/logs`) |
| Windows | `%LOCALAPPDATA%\geo-trax\Logs` |

Each run gets its own file, named after the command module with a timestamp and process ID, e.g. `batch_process_20261023_141502_48213.log`. `--log-path` changes the folder, or pins an exact file when given a path that ends in `.log` or `.txt` (an existing file is used as is, whatever its suffix). `geotrax batch --dry-run` writes no log file.

## Update check

When a command starts, geo-trax asks PyPI in the background, at most once a day, whether a newer release exists, and prints a notice if one does. The request only fetches the package's public release information, and failures (offline, timeout) are silent. To turn it off, set `GEOTRAX_DISABLE_UPDATE_CHECK` to any non-empty value, e.g. `export GEOTRAX_DISABLE_UPDATE_CHECK=1`. Stabilo runs a similar check of its own, which `STABILO_DISABLE_UPDATE_CHECK` turns off.

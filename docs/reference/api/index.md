# API reference

geo-trax is meant to be used through the `geotrax` command, but the helpers that every stage shares can also be imported in your own scripts, as the analysis scripts in [`tools/`](https://github.com/rfonod/geo-trax/blob/main/tools/README.md) do:

| Module | Use it to |
|---|---|
| [`geotrax.utils.registration`](registration.md) | Estimate a homography between two images, e.g. a drone frame and an orthophoto, with any Stabilo detector. |
| [`geotrax.utils.config_utils`](config_utils.md) | Load a pipeline config or preset, apply `--set`-style overrides, and resolve model references and class names. |
| [`geotrax.utils.file_utils`](file_utils.md) | Locate the result files of a video, write files atomically, and read location IDs and video metadata. |
| [`geotrax.utils.logging_utils`](logging_utils.md) | Create the logger that the other helpers expect. |

## Example

Register a video frame to an orthophoto with the same settings as `geotrax georeference`, after a one-off override:

```python
import cv2

from geotrax.utils.config_utils import apply_cli_overrides, load_config
from geotrax.utils.logging_utils import setup_logger
from geotrax.utils.registration import estimate_homography

logger = setup_logger("my_registration", dry_run=True)
cfg = load_config("default", logger)
apply_cli_overrides(cfg, ["matching.downsample_ratio=0.5"], None, logger)

frame, ortho = cv2.imread("frame.png"), cv2.imread("ortho.png")
H, inliers, matches, _ = estimate_homography(frame, ortho, logger, **cfg["georef"]["matching"])
if H is not None:
    print(f"{inliers} inliers out of {matches} matches")
```

## Conventions

- **Loggers.** Most helpers take a `logging.Logger`. Pass one from [`setup_logger`](logging_utils.md#geotrax.utils.logging_utils.setup_logger): some helpers log at the custom NOTICE level, which a plain `logging.getLogger()` logger lacks until `geotrax.utils.logging_utils` has been imported.
- **Fatal errors exit.** The helpers serve a command-line tool, so an unrecoverable input error (a missing config, an invalid override, a failed model download) is logged and ends the process with `SystemExit(1)`. The **Raises** section of each function says when; catch `SystemExit` if your script must carry on.
- **Stability.** Everything documented here is public. Names starting with `_` are left out; they can change in any release.

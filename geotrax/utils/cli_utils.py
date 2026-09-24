# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""Shared CLI argument helpers used by the pipeline entry points."""

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

# Default pipeline config: the bundled 'default' preset. Resolved by resolve_config_path(),
# which falls back to the package's bundled cfg/ dir, so this works from any working directory
# and for both source checkouts and pip-installed wheels.
DEFAULT_CFG = 'geotrax/cfg/default.yaml'


class CfgArg(NamedTuple):
    """How one CLI argument maps onto a key in the pipeline config.

    ``path``     dotted config path, e.g. 'ultralytics.conf' or 'extraction.sahi.enable'.
    ``invert``   the argument holds the negation of the config value (--no-master vs use_master).
    ``no_sync``  record the mapping but reconcile neither direction. For --model and
                 --class-names, whose CLI form is not the config form (ID=NAME tokens against a
                 mapping) and whose precedence ``resolve_class_names``/``resolve_model_path``
                 already implement, including the CLI-vs-config provenance they report. The
                 mapping is still registered so that ``--set`` conflict detection covers them.
    ``coerce``   applied to the config value when it is pulled into the argument (e.g. Path).
    ``flag``     the argument's own long flag (e.g. '--output-folder'), for conflict messages.
                 Not derivable from ``dest`` alone: a parser registered with ``dest_prefix``
                 (``add_plotting_args``) has a ``dest`` that differs from the real flag name.
    """

    path: str
    invert: bool = False
    no_sync: bool = False
    coerce: Callable | None = None
    flag: str = ''


def add_cfg_arg(group, *flags, cfg: str, paths: dict, help: str, default_note: str = '',  # noqa: A002
                invert: bool = False, no_sync: bool = False, coerce: Callable | None = None,
                **kwargs):
    """Register a config-backed CLI argument and record its ``dest -> CfgArg`` mapping.

    The argument always defaults to ``None`` so that "not given on the command line" stays
    distinguishable from "given, and happens to equal the config value"; ``sync_args_with_config``
    (config_utils) then fills it from *cfg* or writes it back, in one place for every stage.

    The "Defaults to cfg -> a -> b." sentence is generated from *cfg* rather than written by hand,
    so the help text cannot drift from the key it documents. Pass *default_note* to append a
    further fallback ("then '<ortho-folder>/master_frames'"), or ``default_note=None`` to suppress
    the generated sentence entirely when the flag needs to phrase it differently.
    """
    help_text = help
    if default_note is not None:
        arrow_path = ' -> '.join(cfg.split('.'))
        help_text = f"{help} Defaults to cfg -> {arrow_path}{', ' + default_note if default_note else ''}."
    action = group.add_argument(*flags, default=None, help=help_text, **kwargs)
    flag = next((f for f in action.option_strings if f.startswith('--')), action.option_strings[0])
    paths[action.dest] = CfgArg(path=cfg, invert=invert, no_sync=no_sync, coerce=coerce, flag=flag)
    return action


def finalize_cli_args(parser, cfg_paths: dict, argv: list | None = None):
    """Parse the command line and stash what the config plumbing needs on the namespace.

    ``_cfg_paths`` is the merged ``dest -> CfgArg`` map, and ``_cli_provided`` is the set of
    config-backed dests actually given on this command line. The latter must be captured here,
    while every such dest is still ``None``: ``sync_args_with_config`` later fills them from the
    config, after which a parsed value is indistinguishable from a typed one. ``--set`` conflict
    detection needs the typed ones, and the config is re-loaded once per stage per video under
    ``geotrax batch``, so every load after the first would otherwise see a full namespace and
    report a conflict against a flag the user never passed.
    """
    parser.set_defaults(_cfg_paths=cfg_paths)
    args = parser.parse_args(argv)
    args._cli_provided = frozenset(dest for dest in cfg_paths if getattr(args, dest, None) is not None)
    return args


def add_common_args(group, cfg: bool = True, output_folder: bool = True) -> dict:
    """
    Register the options shared by all geo-trax commands on the given parser/group:
    ``--cfg`` (unless ``cfg=False``), ``--set``, ``--output-folder`` (unless
    ``output_folder=False``), ``--log-path``, and ``--verbose``.

    Returns the ``dest -> CfgArg`` map for the config-backed options registered here, to be
    merged into the command's ``_cfg_paths`` (see ``add_cfg_arg``).
    """
    paths = {}
    if cfg:
        group.add_argument('--cfg', '-c', type=Path, default=DEFAULT_CFG,
                           help="Pipeline config: a bundled preset name (default, confident, lenient, stable) or a path "
                                "to a custom config file. Run 'geotrax config show' to list presets or 'geotrax config copy' to customize.")
        group.add_argument('--set', '-st', dest='set', action='append', default=None, metavar='KEY=VALUE',
                           help="Override a pipeline config value for this run; repeat for more than one, e.g. "
                                "--set conf=0.35 --set iou=0.6. KEY is a dotted path ('ultralytics.conf') or any "
                                "unambiguous tail of one ('conf'); an ambiguous or unknown key is an error that lists "
                                "the candidates. VALUE is read with YAML rules, so true/null/0.35/[0,1,2] all mean what "
                                "they do in the config file. Use the dedicated flag instead where one exists (it "
                                "validates its input); passing both for the same key is an error.")
    if output_folder:
        add_cfg_arg(group, '--output-folder', '-of', type=str, cfg='output.folder', paths=paths,
                    help="Root folder for pipeline outputs. A bare name (e.g. 'results') creates a sub-folder next to "
                         "each input video; an absolute path is used as-is for all inputs in a batch "
                         "(not recommended when video filenames are not unique across the batch — outputs may overwrite each other). "
                         "Also sets the base under which plots/ is created.",
                    default_note="historical default: 'results'")
    group.add_argument('--log-path', '-lp', type=Path, default=None,
                       help="Where to write detailed logs: a directory (an auto-named "
                            "<stage>_<timestamp>_<pid>.log file is created inside it, so concurrent "
                            "or later runs never share a file) or a full file path. "
                            "Defaults to a platform-specific log directory.")
    group.add_argument('--verbose', '-v', action='store_true', help='Set print verbosity level to INFO (default: WARNING).')
    return paths

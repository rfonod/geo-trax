#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
gen_reference.py - Generate the reference pages of the documentation site.

mkdocs-gen-files runs this script on every build (the gen-files plugin in mkdocs.yml), so each
generated page is rebuilt from the source it documents and cannot drift from it:

- reference/config/: one page per top-level section of geotrax/cfg/default.yaml, with a key /
  default / CLI flag / description table built from the config's own comments, an index page,
  and one page per preset (confident, lenient, stable) listing its changed keys and its full
  diff against default.yaml.
- reference/cli/: one page per 'geotrax' subcommand, rendered from the same argparse parser the
  command uses (each module's ``_build_parser``), with every config-backed flag linked to its
  key through the ``dest -> CfgArg`` map; plus an index built from ``geotrax.cli.COMMANDS``.
- citation.md: built from CITATION.cff.

Drift that would otherwise ship as silently wrong documentation is logged as a warning, which
``mkdocs build --strict`` turns into a failed build: a config leaf without an inline comment, a
'CLI:' hint in a comment naming a flag that no parser maps to that key, a parser mapping a flag
to a key that default.yaml lacks, a README BibTeX entry that no longer matches CITATION.cff, and
a public function, class or documented constant of a module with a page under reference/api/ that
the page does not render (the hand-written API pages list their members one by one, grouped by
topic, so a new helper would otherwise stay undocumented; API_UNLISTED names the deliberate
omissions). A config line the scanner cannot read raises, since every page after it would be wrong.

The scanner reads comments line by line rather than through a YAML library, because PyYAML
drops comments; the values themselves come from ``yaml.safe_load``. A '#' starts a comment
only outside quotes and after whitespace, as in YAML itself, so color strings such as
'#76b041' stay values. Comment lines directly above a key (no blank line in between) are that
key's block comment: above a section or nested mapping they become its introduction, and
inside a mapping they open a new sub-table. A commented-out key (``# key: value  # ...``) is
listed as unset rather than dropped, since its comment says why it is left out.
"""

import argparse
import ast
import difflib
import html
import importlib
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import mkdocs_gen_files
import yaml

from geotrax import CFG_DIR
from geotrax.cli import COMMANDS
from geotrax.config import _PRESET_DESCRIPTIONS

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = 'https://github.com/rfonod/geo-trax'
DEFAULT_CFG = CFG_DIR / 'default.yaml'
PRESETS = ['confident', 'lenient', 'stable']
CONFIG_DIR = Path('reference/config')
CLI_DIR = Path('reference/cli')
API_DIR = ROOT / 'docs' / 'reference' / 'api'
API_UNLISTED = {'logging_utils': {'BColors', 'ColoredFormatter', 'FileFormatter', 'notice'}}

KEY_RE = re.compile(r'^(?P<indent> *)(?P<key>-?[\w.-]+):(?P<rest>.*)$')
COMMENTED_KEY_RE = re.compile(r'^(?P<indent> +)#\s?(?P<key>[a-z_]\w*):\s+(?P<rest>\S.*?\s#\s.*)$')
CLI_TAIL_RE = re.compile(r'\s+—\s+(?:CLI:\s*|set via\s+)(?P<hint>.*)$')
FLAG_RE = re.compile(r'(?<![\w-])(--[\w-]+)')
SECTION_TITLE_RE = re.compile(r'^-{3}\s*(?P<title>.*?)\s*-{3}$')

log = logging.getLogger('mkdocs.plugins.geotrax_reference')


@dataclass
class Entry:
    """One key of the config file, in file order."""

    path: tuple
    raw: str
    comment: str
    cli_hint: str
    block: list = field(default_factory=list)
    commented_out: bool = False
    mapping: bool = False
    line: int = 0


def split_comment(text: str) -> tuple[str, str | None]:
    """Split a YAML line into its code and its comment (``None`` when there is none)."""
    quote = None
    for i, char in enumerate(text):
        if quote:
            if char == quote:
                quote = None
        elif char in '\'"':
            quote = char
        elif char == '#' and (i == 0 or text[i - 1] in ' \t'):
            return text[:i], text[i + 1:].strip()
    return text, None


def split_cli_hint(comment: str) -> tuple[str, str]:
    """Separate a trailing '— CLI: ...' (or '— set via ...') hint from a comment's description."""
    match = CLI_TAIL_RE.search(comment)
    if not match:
        return comment, ''
    return comment[:match.start()].rstrip(), match['hint'].strip()


def scan_config(path: Path) -> list[Entry]:
    """
    Read every key of a config file with its value text, inline comment and block comment.

    Flow sequences continued over several lines are joined, and their comment is taken from
    whichever line carries it. Raises ``ValueError`` on a line that is neither blank, a comment
    nor a ``key: value`` pair.
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    entries, stack, block = [], [], []
    i = 0
    while i < len(lines):
        line, lineno = lines[i], i + 1
        i += 1
        stripped = line.strip()
        if not stripped:
            block = []
            continue
        if stripped.startswith('#'):
            commented = COMMENTED_KEY_RE.match(line)
            if commented and stack:
                indent = len(commented['indent'])
                value, comment = split_comment(commented['rest'])
                comment = comment or ''
                while i < len(lines) and lines[i].strip().startswith('#') and \
                        len(lines[i]) - len(lines[i].lstrip()) > indent:
                    comment += ' ' + lines[i].strip()[1:].strip()
                    i += 1
                parents = tuple(key for ind, key in stack if ind < indent)
                description, hint = split_cli_hint(comment.strip())
                entries.append(Entry(parents + (commented['key'],), value.strip(), description, hint,
                                     block, commented_out=True, line=lineno))
                block = []
                continue
            block.append(line)
            continue
        match = KEY_RE.match(line)
        if not match:
            raise ValueError(f'{path.name}:{lineno}: cannot parse {line!r}')
        indent = len(match['indent'])
        stack = [(ind, key) for ind, key in stack if ind < indent] + [(indent, match['key'])]
        value, comment = split_comment(match['rest'])
        value = value.strip()
        while value.count('[') > value.count(']') and i < len(lines):
            more, more_comment = split_comment(lines[i])
            value += ' ' + more.strip()
            comment = more_comment if more_comment is not None else comment
            i += 1
        description, hint = split_cli_hint(comment or '')
        entries.append(Entry(tuple(key for _, key in stack), value, description, hint, block,
                             line=lineno))
        block = []
    parents = {entry.path[:-1] for entry in entries}
    for entry in entries:
        entry.mapping = entry.path in parents
    return entries


def md(text: str) -> str:
    """Escape config or help text for a Markdown table cell or paragraph."""
    text = html.escape(text, quote=False)
    for char in '\\[]*|':
        text = text.replace(char, '\\' + char)
    return text


def code(text: str) -> str:
    """Format text as an inline code span, widening the fence if the text holds a backtick."""
    fence = '``' if '`' in text else '`'
    return f'{fence}{text}{fence}'


def block_to_md(block: list) -> tuple[str, str]:
    """
    Render a block comment as Markdown, returning ``(title, body)``.

    A '--- Title ---' line becomes the title. Lines indented beyond the comment marker (such as
    the folder tree above 'input:') keep their layout in a text code block; other lines are
    joined into paragraphs, with an empty comment line as a paragraph break.
    """
    title, out, para, pre = '', [], [], []

    def flush():
        if para:
            out.append(md(' '.join(para)))
            para.clear()
        if pre:
            out.append('```text\n' + textwrap.dedent('\n'.join(pre)) + '\n```')
            pre.clear()

    for line in block:
        text = line.strip()[1:]
        text = text[1:] if text.startswith(' ') else text
        heading = SECTION_TITLE_RE.match(text.strip())
        if heading and not title and not out and not para:
            title = heading['title']
        elif text.startswith('  ') or any(char in text for char in '├└│'):
            if para:
                flush()
            pre.append(text)
        elif not text.strip():
            flush()
        else:
            if pre:
                flush()
            para.append(text.strip())
    flush()
    return title, '\n\n'.join(out)


def is_short_heading(block: list) -> bool:
    """A one-line block comment without a full stop reads as a sub-heading, not as prose."""
    return len(block) == 1 and len(block[0].strip()) <= 82 and not block[0].rstrip().endswith('.')


def flatten(cfg, prefix: tuple = ()) -> dict:
    """Map every leaf path of a loaded config to its value; lists count as leaves."""
    if isinstance(cfg, dict) and cfg:
        out = {}
        for key, value in cfg.items():
            out |= flatten(value, prefix + (str(key),))
        return out
    return {prefix: cfg}


def show_value(value) -> str:
    """Render a loaded config value the way it would be written in the config file."""
    if isinstance(value, str):
        return repr(value) if value == '' else value
    return yaml.safe_dump(value, default_flow_style=True, width=1000).strip().removesuffix('...').strip()


def anchor(path: tuple) -> str:
    """HTML id of a key's table row, linked to from the CLI and preset pages."""
    return '.'.join(path)


def config_link(path: tuple, from_dir: Path) -> str:
    """Relative link to a key's row on its section page, as seen from a page in *from_dir*."""
    page = os.path.relpath(CONFIG_DIR / f'{path[0]}.md', from_dir).replace(os.sep, '/')
    return f'[{code(".".join(path))}]({page}#{anchor(path)})'


def load_commands() -> dict:
    """Build every subcommand's parser and ``dest -> CfgArg`` map, keyed by command name."""
    os.environ['COLUMNS'] = '100'
    commands = {}
    for name, (module_name, _) in COMMANDS.items():
        parser, cfg_paths = importlib.import_module(module_name)._build_parser()
        if parser.prog != f'geotrax {name}':
            parser.prog = f'geotrax {name}'
        commands[name] = (parser, cfg_paths)
    return commands


def iter_actions(parser: argparse.ArgumentParser):
    """Yield every action of a parser and of its subparsers."""
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from iter_actions(sub)


def flags_by_config_path(commands: dict) -> dict:
    """Map each config path to the option strings that override it, across all subcommands, in parser order."""
    flags = {}
    for parser, cfg_paths in commands.values():
        for action in iter_actions(parser):
            if action.dest in cfg_paths:
                path = tuple(cfg_paths[action.dest].path.split('.'))
                flags.setdefault(path, {}).update(dict.fromkeys(action.option_strings))
    return flags


def check_drift(entries: list, flags: dict) -> None:
    """Warn about comments and parser mappings that disagree with each other or with the config."""
    leaves = {entry.path: entry for entry in entries if not entry.mapping}
    for entry in leaves.values():
        where = f'{DEFAULT_CFG.name}:{entry.line} ({anchor(entry.path)})'
        if not entry.comment and not entry.commented_out:
            log.warning(f'{where} has no inline comment to document it')
        for flag in FLAG_RE.findall(entry.cli_hint):
            if flag != '--set' and flag not in flags.get(entry.path, {}):
                log.warning(f"{where} names CLI flag '{flag}', but no subcommand maps it to this key")
        if entry.path in flags and not entry.cli_hint:
            listed = ', '.join(f for f in flags[entry.path] if f.startswith('--'))
            log.warning(f'{where} is overridden by {listed}, but its comment has no CLI hint')
    for path in flags:
        if path not in leaves:
            log.warning(f"a CLI flag maps to config key '{anchor(path)}', which {DEFAULT_CFG.name} does not have")


def public_members(module_path: Path) -> set:
    """Return the public functions and classes of a module, and its constants that have a docstring."""
    body = ast.parse(module_path.read_text(encoding='utf-8')).body
    names = set()
    for node, following in zip(body, body[1:] + [None], strict=True):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            documented = (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            )
            if isinstance(target, ast.Name) and documented:
                names.add(target.id)
    return {name for name in names if not name.startswith('_')}


def check_api_pages() -> None:
    """Warn about public members of a module with an API page that the page does not render."""
    for page in sorted(API_DIR.glob('*.md')):
        if page.name == 'index.md':
            continue
        module = f'geotrax.utils.{page.stem}'
        rendered = set(re.findall(rf'^::: {re.escape(module)}\.(\w+)\s*$', page.read_text(encoding='utf-8'), re.M))
        source = ROOT / Path(*module.split('.')).with_suffix('.py')
        for name in sorted(public_members(source) - rendered - API_UNLISTED.get(page.stem, set())):
            log.warning(f"reference/api/{page.name} does not render the public member '{module}.{name}'")


def cli_cell(entry: Entry, flags: dict) -> str:
    """The CLI column of a config row: the parsers' long flags, else the comment's own hint."""
    long_flags = [f for f in flags.get(entry.path, ()) if f.startswith('--')]
    if long_flags:
        return ', '.join(code(flag) for flag in long_flags)
    return code(entry.cli_hint) if entry.cli_hint else ''


def render_section(section: str, entries: list, flags: dict) -> str:
    """Render one top-level config section as a page of tables, one per mapping and sub-heading."""
    head = entries[0]
    title, intro = block_to_md(head.block)
    lines = [f'# `{section}`', '']
    if title:
        lines += [f'**{md(title)}.**', '']
    if intro:
        lines += [intro, '']
    lines += [f'Generated from [`geotrax/cfg/default.yaml`]({REPO_URL}/blob/main/geotrax/cfg/default.yaml); '
              'every row below is a key of that file with its shipped default and its inline comment.', '']

    table_open = False
    current_parent = (section,)
    for entry in entries[1:]:
        depth = len(entry.path) - 1
        if entry.mapping:
            level = '#' * min(depth + 1, 4)
            lines += ['', f'{level} `{".".join(entry.path)}`', '']
            _, body = block_to_md(entry.block)
            if body:
                lines += [body, '']
            if entry.comment:
                lines += [md(entry.comment), '']
            table_open = False
            current_parent = entry.path
            continue
        if entry.block:
            _, body = block_to_md(entry.block)
            if is_short_heading(entry.block):
                level = '#' * min(len(entry.path), 5)
                lines += ['', f'{level} {body}', '']
            elif body:
                lines += ['', body, '']
            table_open = False
        if entry.path[:-1] != current_parent:
            lines += ['', f'{"#" * min(len(entry.path), 4)} `{".".join(entry.path[:-1])}` (continued)', '']
            current_parent = entry.path[:-1]
            table_open = False
        if not table_open:
            lines += ['| Key | Default | CLI | Description |', '|---|---|---|---|']
            table_open = True
        key = '.'.join(entry.path[len(current_parent):])
        key_cell = f'<a id="{anchor(entry.path)}"></a>{code(key)}'
        if entry.commented_out:
            default = f'unset (commented out; {code(entry.raw)} when enabled)'
        else:
            default = code(' '.join(entry.raw.split()) if entry.raw else 'null')
        lines.append(f'| {key_cell} | {default} | {cli_cell(entry, flags)} | {md(entry.comment)} |')
    return '\n'.join(lines).replace('\n\n\n', '\n\n') + '\n'


def render_config_index(sections: dict) -> str:
    """Render the config reference landing page: the sections and the presets."""
    lines = [
        '# Config reference',
        '',
        'geo-trax is driven by one self-contained YAML file. These pages are generated from the bundled '
        f'[`default.yaml`]({REPO_URL}/blob/main/geotrax/cfg/default.yaml) on every documentation build, so '
        'they always list the keys, defaults and comments of the shipped file. The CLI column names the '
        'flag that overrides a key for one run; any key can also be overridden with `--set KEY=VALUE`. '
        'See the [Configuration guide](../../guides/configuration.md) for presets, `geotrax config copy` '
        'and the `--set` rules.',
        '',
        '## Sections',
        '',
        '| Section | Purpose |',
        '|---|---|',
    ]
    for section, entries in sections.items():
        _, intro = block_to_md(entries[0].block)
        first = intro.split('\n\n')[0].split('. ')[0].rstrip('.') if intro and not intro.startswith('```') else ''
        lines.append(f'| [`{section}`]({section}.md) | {first} |')
    lines += ['', '## Presets', '',
              'Each preset is a complete config file that differs from `default.yaml` only in a few tuned values. '
              'Select one by name with `-c`, e.g. `geotrax batch video.mp4 -c confident`.', '',
              '| Preset | Focus |', '|---|---|',
              f'| `default` | {md(_PRESET_DESCRIPTIONS["default"])} |']
    lines += [f'| [`{name}`](presets/{name}.md) | {md(_PRESET_DESCRIPTIONS[name])} |' for name in PRESETS]
    return '\n'.join(lines) + '\n'


def render_preset(name: str, default_cfg: dict) -> str:
    """Render one preset as its changed keys plus the full unified diff against default.yaml."""
    preset_file = CFG_DIR / f'{name}.yaml'
    preset_cfg = yaml.safe_load(preset_file.read_text(encoding='utf-8'))
    base, tuned = flatten(default_cfg), flatten(preset_cfg)
    changed = [path for path in dict.fromkeys([*base, *tuned]) if base.get(path, KeyError) != tuned.get(path, KeyError)]
    lines = [f'# `{name}` preset', '', md(_PRESET_DESCRIPTIONS[name]), '',
             f'Use it with `-c {name}`, or start from a local copy with `geotrax config copy`. '
             f'Generated from [`{name}.yaml`]({REPO_URL}/blob/main/geotrax/cfg/{name}.yaml).', '',
             '## Changed keys', '', '| Key | `default` | `' + name + '` |', '|---|---|---|']
    preset_dir = CONFIG_DIR / 'presets'
    for path in changed:
        old = code(show_value(base[path])) if path in base else 'absent'
        new = code(show_value(tuned[path])) if path in tuned else 'absent'
        key = config_link(path, preset_dir) if path in base else code('.'.join(path))
        lines.append(f'| {key} | {old} | {new} |')
    diff = difflib.unified_diff(DEFAULT_CFG.read_text(encoding='utf-8').splitlines(),
                                preset_file.read_text(encoding='utf-8').splitlines(),
                                'default.yaml', f'{name}.yaml', lineterm='', n=1)
    lines += ['', '??? note "Full diff against `default.yaml`"', '', '    ```diff']
    lines += [f'    {line}' if line else '' for line in diff]
    lines += ['    ```']
    return '\n'.join(lines) + '\n'


def value_cell(action: argparse.Action) -> str:
    """The value an option takes: its choices, its metavar (with repetition), or nothing for a switch."""
    if action.nargs == 0:
        return ''
    if action.choices and not isinstance(action, argparse._SubParsersAction):
        return ' \\| '.join(code(str(choice)) for choice in action.choices)
    if not action.option_strings:
        return ''
    metavar = action.metavar or action.dest.upper()
    if isinstance(metavar, tuple):
        metavar = ' '.join(metavar)
    repeated = {'+': f'{metavar} [{metavar} ...]', '*': f'[{metavar} ...]', '?': f'[{metavar}]'}
    return code(repeated.get(action.nargs, metavar))


def help_cell(action: argparse.Action, cfg_paths: dict) -> str:
    """An option's help text, with argparse's %-placeholders filled and a fixed default noted."""
    text = action.help or ''
    if '%(' in text:
        text = text % {**vars(action), 'prog': ''}
    if (action.dest not in cfg_paths and not any(action.default is v for v in (None, False, argparse.SUPPRESS))
            and 'default' not in text.lower()):
        text += f' Default: {action.default}.'
    return md(text.strip())


def render_parser(parser: argparse.ArgumentParser, cfg_paths: dict, level: int) -> list:
    """Render a parser's usage and argument groups as Markdown, recursing into subcommands."""
    lines = ['```text', parser.format_usage().strip(), '```', '']
    if parser.description:
        lines += [md(parser.description), '']
    subcommands = []
    for group in parser._action_groups:
        actions = [a for a in group._group_actions if not isinstance(a, argparse._HelpAction)]
        subcommands += [a for a in actions if isinstance(a, argparse._SubParsersAction)]
        actions = [a for a in actions if not isinstance(a, argparse._SubParsersAction)]
        if not actions:
            continue
        lines += [f'{"#" * (level + 1)} {group.title[0].upper()}{group.title[1:]}', '']
        if group.description:
            lines += [md(group.description), '']
        lines += ['| Argument | Value | Config key | Description |', '|---|---|---|---|']
        for action in actions:
            name = ', '.join(code(flag) for flag in action.option_strings) or code(action.dest)
            cfg_arg = cfg_paths.get(action.dest)
            key = config_link(tuple(cfg_arg.path.split('.')), CLI_DIR) if cfg_arg else ''
            if cfg_arg and cfg_arg.invert:
                key += ' (negated)'
            lines.append(f'| {name} | {value_cell(action)} | {key} | {help_cell(action, cfg_paths)} |')
        lines.append('')
    for action in subcommands:
        for sub in action.choices.values():
            lines += [f'{"#" * (level + 1)} `{sub.prog}`', '']
            lines += render_parser(sub, {}, level + 1)
    return lines


def render_command(name: str, parser: argparse.ArgumentParser, cfg_paths: dict) -> str:
    """Render one subcommand's page."""
    lines = [f'# `geotrax {name}`', '', md(COMMANDS[name][1]) + '.', '']
    lines += render_parser(parser, cfg_paths, level=1)
    return '\n'.join(lines) + '\n'


def render_cli_index() -> str:
    """Render the CLI reference landing page from the subcommand registry."""
    lines = ['# CLI reference', '',
             'Installing geo-trax provides the `geotrax` command (`python -m geotrax` is equivalent). '
             'These pages are generated from the argparse parsers of the subcommands on every documentation '
             'build. Where a flag overrides a config key, the Config key column links to that key; the flag '
             'wins over the config file for that one run.', '',
             '| Command | Purpose |', '|---|---|']
    lines += [f'| [`geotrax {name}`]({name}.md) | {md(description)} |' for name, (_, description) in COMMANDS.items()]
    return '\n'.join(lines) + '\n'


def bibtex_article(cff: dict) -> str:
    """The journal article's BibTeX entry, from CITATION.cff's preferred-citation."""
    ref = cff['preferred-citation']
    authors = ' and '.join(f'{a["family-names"]}, {a["given-names"]}' for a in ref['authors'])
    key = f'{ref["authors"][0]["family-names"].lower()}{ref["year"]}{ref["title"].split()[0].lower()}'
    fields = [('title', ref['title'].strip()), ('author', authors), ('journal', ref['journal']),
              ('volume', ref['volume']), ('pages', ref['start']), ('year', ref['year']),
              ('publisher', ref['publisher']['name']), ('doi', ref['doi']), ('url', ref['url'])]
    return f'@article{{{key},\n' + ',\n'.join(f'  {k} = {{{v}}}' for k, v in fields) + '\n}'


def bibtex_software(cff: dict) -> str:
    """The software archive's BibTeX entry, from CITATION.cff's top level."""
    released = str(cff['date-released'])
    year, month = released[:4], int(released[5:7])
    author = cff['authors'][0]
    repo = cff['repository-code'].rstrip('/').rsplit('/', 1)[-1]
    months = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()
    doi = next(i['value'] for i in cff['identifiers'] if i['type'] == 'doi')
    fields = [('author', f'{{{author["family-names"]}, {author["given-names"]}}}'), ('title', f'{{{cff["title"]}}}'),
              ('year', f'{{{year}}}'), ('month', months[month - 1]), ('version', f'{{{cff["version"]}}}'),
              ('license', f'{{{cff["license"]}}}'), ('doi', f'{{{doi}}}'), ('url', f'{{{cff["repository-code"]}}}')]
    key = f'{author["family-names"].lower()}{year}{repo}'
    return f'@software{{{key},\n' + ',\n'.join(f'  {k} = {v}' for k, v in fields) + '\n}'


def render_citation() -> str:
    """Render the citation page, and warn if the README's BibTeX has drifted from CITATION.cff."""
    cff = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    article, software = bibtex_article(cff), bibtex_software(cff)
    readme_entries = re.findall(r'```bibtex\n(.*?)\n```', (ROOT / 'README.md').read_text(encoding='utf-8'), re.S)
    for entry in (article, software):
        if entry not in readme_entries:
            log.warning(f"README.md's BibTeX for '{entry.splitlines()[0]}' does not match CITATION.cff")
    ref = cff['preferred-citation']
    doi = next(i['value'] for i in cff['identifiers'] if i['type'] == 'doi')
    lines = [
        '# Citation', '',
        f'{md(cff["message"])} Both entries are generated from [`CITATION.cff`]({REPO_URL}/blob/main/CITATION.cff), '
        'which GitHub also offers under "Cite this repository".', '',
        '## Journal article', '',
        'Preferred for any use of the framework. The article is open access.', '',
        f'> {md(ref["title"].strip())}. *{md(ref["journal"])}* {ref["volume"]}, {ref["start"]} ({ref["year"]}). '
        f'[doi:{ref["doi"]}](https://doi.org/{ref["doi"]})', '',
        '```bibtex', article, '```', '',
        '## Software archive', '',
        f'When referencing or building on the code itself. The Zenodo DOI [{doi}](https://doi.org/{doi}) always '
        f'resolves to the latest release; the entry below records version {cff["version"]}.', '',
        '```bibtex', software, '```', '',
        '## Datasets', '',
        'Released on Zenodo alongside the article:', '',
        '- **Songdo Traffic**: about 700,000 georeferenced vehicle trajectories produced with geo-trax, '
        '[doi:10.5281/zenodo.13828383](https://doi.org/10.5281/zenodo.13828383).',
        '- **Songdo Vision v1**: the annotated aerial images the default detector was trained on, '
        '[doi:10.5281/zenodo.13828408](https://doi.org/10.5281/zenodo.13828408).',
        '- **Songdo Vision v2**: an annotation-quality release of the same images, '
        '[doi:10.5281/zenodo.21728430](https://doi.org/10.5281/zenodo.21728430).',
    ]
    return '\n'.join(lines) + '\n'


def write(path: Path, text: str, edit_source: str) -> None:
    """Write one generated page and point its 'edit' button at the file it is generated from."""
    with mkdocs_gen_files.open(path.as_posix(), 'w') as handle:
        handle.write(text)
    mkdocs_gen_files.set_edit_path(path.as_posix(), f'../{edit_source}')


def main() -> None:
    """Generate every reference page."""
    entries = scan_config(DEFAULT_CFG)
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding='utf-8'))
    commands = load_commands()
    flags = flags_by_config_path(commands)
    check_drift(entries, flags)
    check_api_pages()

    sections = {}
    for entry in entries:
        sections.setdefault(entry.path[0], []).append(entry)
    for section, section_entries in sections.items():
        write(CONFIG_DIR / f'{section}.md', render_section(section, section_entries, flags), 'geotrax/cfg/default.yaml')
    write(CONFIG_DIR / 'index.md', render_config_index(sections), 'geotrax/cfg/default.yaml')
    for name in PRESETS:
        write(CONFIG_DIR / 'presets' / f'{name}.md', render_preset(name, cfg), f'geotrax/cfg/{name}.yaml')

    for name, (parser, cfg_paths) in commands.items():
        module = COMMANDS[name][0].replace('.', '/') + '.py'
        write(CLI_DIR / f'{name}.md', render_command(name, parser, cfg_paths), module)
    write(CLI_DIR / 'index.md', render_cli_index(), 'geotrax/cli.py')
    write(Path('citation.md'), render_citation(), 'CITATION.cff')


main()

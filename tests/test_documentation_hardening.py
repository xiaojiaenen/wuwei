"""Execute marked public examples; validate links/imports without network access."""
import importlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = [ROOT / p for p in ('README.md', 'USAGE.md', 'MIGRATION.md', 'CHANGELOG.md')]
PUBLIC += [p for p in (ROOT / 'docs').rglob('*.md') if not {'audits', 'superpowers'} & set(p.parts)]


def test_offline_documentation_examples():
    source = (ROOT / 'docs/examples/offline.md').read_text()
    blocks = re.findall(r'```python\n(.*?)```', source, re.S)
    assert len(blocks) >= 4
    for index, block in enumerate(blocks):
        exec(compile(block, f'offline.md:example{index}', 'exec'), {'__name__': '__documentation__'})


def test_public_import_examples():
    import ast
    for path in PUBLIC:
        for block in re.findall(r'```python[^\n]*\n(.*?)```', path.read_text(), re.S):
            tree = ast.parse(block)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('wuwei'):
                    module = importlib.import_module(node.module)
                    for alias in node.names:
                        assert hasattr(module, alias.name), (path, node.module, alias.name)


def test_public_markdown_links_and_removed_api_examples():
    forbidden = ('PluginLoader', 'PluginRegistry', 'graph.channels', 'SkillViewerTool',
                 'Tool.from_function', '业界无开源对标')
    for path in PUBLIC:
        source = path.read_text()
        for old in forbidden:
            assert old not in source, (path, old)
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)', source):
            if '://' in target or target.startswith(('#', 'mailto:')):
                continue
            local = target.split('#', 1)[0]
            assert (path.parent / local).exists(), (path, target)


def test_mkdocs_navigation_targets():
    source = (ROOT / 'mkdocs.yml').read_text().split('nav:', 1)[1]
    for target in re.findall(r':\s+([^\s]+\.md)', source):
        assert (ROOT / 'docs' / target).is_file(), target

"""Syntax and forbidden accelerator-extension imports; no torch needed."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    files, errors = [], []
    forbidden = {'torch_sparse', 'torch_scatter', 'torch_geometric', 'cupy', 'triton', 'flash_attn', 'apex'}
    for directory in ('pfn_network', 'scripts', 'tests'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            text = path.read_text(encoding='utf-8')
            tree = ast.parse(text, filename=str(path), feature_version=(3, 10))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    imports.add(node.module or '')
            relative = path.relative_to(ROOT).as_posix()
            files.append({'file': relative, 'imports': sorted(imports)})
            if directory == 'pfn_network':
                if any(m.split('.')[0] in forbidden for m in imports):
                    errors.append(f'{relative}: forbidden extension import')
                if re.search(r'torch\.cuda|\.cuda\(|torch\.compile|["\x27]nccl["\x27]', text):
                    errors.append(f'{relative}: accelerator-specific call found')
    print(json.dumps({'python_grammar': '3.10', 'files': files, 'errors': errors}, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()

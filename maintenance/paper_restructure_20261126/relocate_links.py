"""Relocate local Markdown links without editing scientific JSON locks."""
import json
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


def relocated_link(old, new, link, mapping):
    parsed=urlsplit(link)
    if parsed.scheme or parsed.netloc or not parsed.path or parsed.path.startswith('/'):
        return link
    source=posixpath.normpath(posixpath.join(posixpath.dirname(old),unquote(parsed.path)))
    target=mapping.get(source)
    if target is None:
        return link
    target=quote(posixpath.relpath(target,posixpath.dirname(new) or '.'),safe='/._-')
    return target+('?' + parsed.query if parsed.query else '')+('#' + parsed.fragment if parsed.fragment else '')


def run(root=Path.cwd()):
    here=root/'maintenance/paper_restructure_20261126'
    mapping=json.loads((here/'path_mapping.json').read_text(encoding='utf-8'))['paths']
    changed=[]
    for old,new in mapping.items():
        if not new.endswith('.md') or not (root/new).exists(): continue
        path=root/new;original=path.read_text(encoding='utf-8')
        updated=re.sub(r'(\]\()([^\n)]+)(\))',lambda m:m[1]+relocated_link(old,new,m[2],mapping)+m[3],original)
        if original!=updated:
            path.write_text(updated,encoding='utf-8',newline='\n');changed.append(new)
    (here/'link_relocations.json').write_text(json.dumps(dict(changed_markdown=changed,policy='local link targets only; scientific locks and numeric results unchanged'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(markdown_updated=len(changed))))


if __name__=='__main__': run()

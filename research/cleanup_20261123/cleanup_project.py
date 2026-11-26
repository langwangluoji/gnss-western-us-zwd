"""Approved one-time cleanup; paths checked, no reparse traversal, durable audit."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import shutil
import sys
import time

_root = Path(__file__).resolve().parents[2]
# Historical artifacts exceed MAX_PATH; use native extended paths throughout.
ROOT = Path('\\\\?\\' + str(_root)) if os.name == 'nt' else _root
AUDIT = ROOT / 'maintenance/cleanup_20261123'
ARCHIVE = ROOT.parent / 'GAN对流层_历史归档_20261123'
PIPE = ROOT / 'research_pipeline'
STUDY = PIPE / 'studies/us_dense_zwd_20261105_v1'
PROJECT = STUDY / '03_code/gnss_field_server'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def files(path):
    if path.is_file():
        yield path
        return
    for base, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base)/d).is_junction() and not (Path(base)/d).is_symlink()]
        for name in names:
            p = Path(base)/name
            if not p.is_symlink():
                yield p


def size(path):
    return sum(p.stat().st_size for p in files(path))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(path, root):
    p = path.absolute()
    if p == root or not p.is_relative_to(root):
        raise ValueError(f'Unsafe path: {p}')
    # Do not resolve a link into a different target.
    for parent in [p, *p.parents]:
        if parent == root:
            break
        if parent.is_symlink() or parent.is_junction():
            raise ValueError(f'Reparse point in operation: {parent}')
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {p}')
    return p


def import_closure(seeds, roots):
    modules = {}
    for base in roots:
        for p in base.glob('*.py'):
            modules.setdefault(p.stem, p)
    todo = list(seeds)
    retained = set()
    while todo:
        p = todo.pop()
        if p in retained or not p.is_file():
            continue
        retained.add(p)
        try:
            tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import): names = [v.name for v in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module: names = [node.module]
            for name in names:
                target = modules.get(name.split('.')[0])
                if target and target not in retained: todo.append(target)
    return retained


def make_plan():
    AUDIT.mkdir(parents=True, exist_ok=True)
    with (AUDIT/'inventory_before.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f); writer.writerow(['path', 'bytes'])
        for p in files(ROOT):
            if p.is_relative_to(AUDIT): continue
            writer.writerow([p.relative_to(ROOT).as_posix(), p.stat().st_size])
    ops = []
    def move(path, reason):
        if path.exists() and not path.is_junction() and not path.is_symlink():
            ops.append(dict(action='archive', path=path.relative_to(ROOT).as_posix(), bytes=size(path), reason=reason))
    for p in ROOT.iterdir():
        if p.is_junction() or p.is_symlink():
            ops.append(dict(action='remove_link',path=p.relative_to(ROOT).as_posix(),bytes=0,
                            reason='Remove obsolete compatibility alias only; target is not traversed',target=str(p.readlink())))
            continue
        if p.name not in {'research_pipeline', 'docs', 'maintenance', 'README.md', 'AGENTS.md', '.gitignore', '.git'}:
            if p.name == '.pytest_cache': continue
            move(p, 'Historical material outside the current western-US research')
    for p in (PIPE/'studies').iterdir():
        if p.name not in {STUDY.name, 'us_dense_zwd_20261105_plan', 'README.md'}:
            move(p, 'Historical research region')
    boundary = PIPE/'studies/us_dense_zwd_20261105_plan'
    for p in boundary.iterdir():
        if p.name != 'evidence': move(p, 'Old region planning material')
    for p in (boundary/'evidence').iterdir():
        if p.name != 'countries.geojson': move(p, 'Only country boundaries needed for current geometry')
    for p in PIPE.iterdir():
        if p.name not in {'studies','data','scripts','src','README.md','.gitignore','.pytest_cache'}:
            move(p, 'Early shared experiments; current engineering is self-contained')
    for p in (PIPE/'data').iterdir():
        if p.name not in {'pwv_data','external'}: move(p, 'Legacy data outside current region')
    for p in (PIPE/'data/external').iterdir():
        if p.name not in {'egm96_15.gtx','us_nga_egm96_15.tif'}:
            move(p, 'Legacy external data outside current dependencies')
    code = STUDY/'03_code'
    seeds = list((code/'field_generation_v1').glob('*.py'))
    seeds += list(PROJECT.glob('src/**/*.py')) + list(PROJECT.glob('tools/*.py'))
    seeds += [code/'pwv_local.py',code/'acquire_srtm.py']
    closure = import_closure(seeds, [code, PIPE/'scripts'])
    for p in code.iterdir():
        if p.name in {'gnss_field_server','field_generation_v1','residual_edm_v1','__pycache__'}: continue
        if p not in closure: move(p, 'No import dependency from the retained current preparation/runtime')
    for p in (PIPE/'scripts').iterdir():
        if p.name != '__pycache__' and p not in closure: move(p, 'No import dependency from current source parser')
    for p in STUDY.iterdir():
        if p.name in {'00_protocol','07_operations','08_delivery'}: move(p, 'Superseded old-region protocols and operations')
    for p in (STUDY/'04_models').iterdir():
        if p.name != 'field_generation_v1': move(p, 'Old 39-station regional model; not current expanded-region progress')
    for p in (STUDY/'02_data/processed').iterdir():
        if p.name not in {'field_generation_v1','gnss_field_server_v1'}: move(p, 'Superseded small-region derived data')
    for p in (STUDY/'02_data/raw').iterdir():
        if p.name != 'srtm': move(p, 'Superseded old-region raw download')
    for p in (STUDY/'06_reports').iterdir():
        if p.name not in {'data_distribution_20261116','field_metrics_pro_20261117'}:
            move(p, 'Superseded small-region report')
    for p in (STUDY/'01_inventory').iterdir():
        if p.name not in {'pwv_local','srtm_manifest.json'}: move(p, 'Superseded regional station inventory')
    for p in (STUDY/'01_inventory/pwv_local').iterdir():
        if p.name != 'global_station_headers.csv': move(p, 'Only global source inventory is used by current preparation')
    # Preserve original western-US candidate observations; move all other global sources.
    with (STUDY/'01_inventory/pwv_local/global_station_headers.csv').open(encoding='utf-8-sig', newline='') as f:
        sources = {Path(r['source_path']).name.lower() for r in csv.DictReader(f)
                   if 25 <= float(r['lat']) <= 50 and -125 <= float(r['lon']) <= -110}
    for p in (PIPE/'data/pwv_data').iterdir():
        if p.name.lower() not in sources: move(p, 'Original observations outside current western-US candidate bounds')
    # Staging copies and release packages stay recoverable outside the active project.
    move(PROJECT/'dist', 'Recoverable release staging; source and scientific results retained')
    for p in (PROJECT/'build').iterdir():
        if p.name not in {'full_generative_real_check_v01','full_generative_real_check_v02','full_generative_real_check_v03'}:
            move(p, 'Recoverable package staging; actual full-generative validation runs retained separately')
    # Delete complete cache directory, not isolated arrays with an intact committed cursor.
    for rel, generator in [
        ('20261121_grid_processor_pair_v01/scientific/cache','tools/grid_pair_data.py: GridData.prepare; tools/grid_pair_train.py'),
        ('20261122_local_generative_architecture_v01/cache','results/20261122_local_generative_architecture_v01/pilot_data.py: prepare_cache')]:
        p = PROJECT/'results'/rel
        if p.exists():
            ops.append(dict(action='delete_cache', path=p.relative_to(ROOT).as_posix(), bytes=size(p),
                            reason='Regenerable frozen input features; complete directory reset including cursor', reconstruction=generator))
    # Deduplicate bytecode/test caches; skip subtrees already moved/deleted above.
    scheduled = {ROOT/o['path'] for o in ops}
    for base, dirs, _ in os.walk(ROOT, followlinks=False):
        for name in list(dirs):
            p = Path(base)/name
            if p.is_junction() or p.is_symlink() or any(q in scheduled for q in (p,*p.parents)):
                dirs.remove(name); continue
            if name in {'__pycache__','.pytest_cache'}:
                ops.append(dict(action='delete_cache', path=p.relative_to(ROOT).as_posix(), bytes=size(p),
                                reason='Python bytecode or pytest cache', reconstruction='Automatically rebuilt by Python/pytest'))
                scheduled.add(p); dirs.remove(name)
    # Ensure all operations are disjoint and contained before doing anything.
    paths = set()
    for o in sorted(ops,key=lambda o:len(Path(o['path']).parts)):
        if o['action']=='remove_link':
            p=ROOT/o['path']
            if p.parent!=ROOT or not (p.is_junction() or p.is_symlink()):raise ValueError('Invalid link operation')
        else:
            p = safe(ROOT/o['path'], ROOT)
        if any(q in paths for q in (p,*p.parents)):
            raise ValueError(f'Overlapping operations: {p}')
        paths.add(p)
    summary = dict(operations=len(ops), archive_bytes=sum(o['bytes'] for o in ops if o['action']=='archive'),
                   delete_bytes=sum(o['bytes'] for o in ops if o['action']=='delete_cache'),
                   retained_support_modules=sorted(str(p.relative_to(ROOT)) for p in closure if p.is_relative_to(ROOT)),
                   current_original_source_files=len(sources), archive=str(ARCHIVE))
    write(AUDIT/'operations.json',ops); write(AUDIT/'plan_summary.json', summary)
    # Scientific artifacts to verify unchanged, including data, code, weight, and metric identities.
    protected = []
    roots = [STUDY/'02_data/processed/gnss_field_server_v1', STUDY/'02_data/processed/field_generation_v1', PROJECT/'src', PROJECT/'tools', PROJECT/'results', PROJECT/'build']
    for r in roots:
        for p in files(r):
            if any(q in paths for q in (p,*p.parents)): continue
            if r == PROJECT/'results' and p.suffix not in {'.pt','.pth','.json','.py','.md'}: continue
            if '__pycache__' in p.parts: continue
            protected.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p),bytes=p.stat().st_size))
    write(AUDIT/'protected_artifacts_before.json', protected)
    print(json.dumps(summary, ensure_ascii=True))


def execute():
    ops = json.loads((AUDIT/'operations.json').read_text(encoding='utf-8'))
    if ARCHIVE.is_junction() or ARCHIVE.is_symlink():raise ValueError('Archive root is a reparse point')
    ARCHIVE.mkdir(exist_ok=True)
    ledger = AUDIT/'operations_completed.jsonl'
    completed = {json.loads(x)['path'] for x in ledger.read_text(encoding='utf-8').splitlines()} if ledger.exists() else set()
    with ledger.open('a', encoding='utf-8') as log:
        for o in ops:
            if o['path'] in completed: continue
            if o['action']=='remove_link':
                p=ROOT/o['path']
                if p.parent!=ROOT or not (p.is_junction() or p.is_symlink()):raise ValueError('Invalid link operation')
                if p.is_junction():os.rmdir(p)
                else:p.unlink()
                log.write(json.dumps({**o,'completed':time.time()},ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
                continue
            p = safe(ROOT/o['path'], ROOT)
            if not p.exists(): raise ValueError(f'Source missing without receipt: {p}')
            if size(p) != o['bytes']: raise ValueError(f'Source changed: {p}')
            if o['action']=='archive':
                dest = safe(ARCHIVE/o['path'], ARCHIVE)
                if dest.exists(): raise ValueError(f'Archive collision: {dest}')
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(p), str(dest))
                if size(dest) != o['bytes']: raise ValueError(f'Archive size mismatch: {dest}')
            else:
                # Save cache metadata before reset; file paths and sizes already in original inventory.
                for meta in files(p):
                    if meta.suffix in {'.json','.jsonl'}:
                        out=AUDIT/'deleted_cache_metadata'/meta.relative_to(ROOT)
                        out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(meta,out)
                shutil.rmtree(p)
            log.write(json.dumps({**o,'completed':time.time()},ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
    (ARCHIVE/'README.md').write_text('# 历史研究归档\n\n由用户于2026-11-23授权移出活动项目。相对目录结构保留；移动清单见原项目 maintenance/cleanup_20261123/operations_completed.jsonl。这里包含旧区域研究、历史汇报、非当前区域原始站点数据和工程发行副本。\n',encoding='utf-8')
    print(json.dumps({'completed':len(ops),'archive':str(ARCHIVE)},ensure_ascii=True))


def verify():
    protected=json.loads((AUDIT/'protected_artifacts_before.json').read_text(encoding='utf-8'))
    failures=[]
    for row in protected:
        p=ROOT/row['path']
        if not p.exists() or digest(p)!=row['sha256']:failures.append(row['path'])
    ops=json.loads((AUDIT/'operations.json').read_text(encoding='utf-8'))
    restoration=AUDIT/'retained_compatibility_files.json'
    restored=json.loads(restoration.read_text(encoding='utf-8')) if restoration.exists() else []
    allowed={x['path']:x['sha256'] for x in restored}
    for rel,expected in allowed.items():
        if not (ROOT/rel).is_file() or digest(ROOT/rel)!=expected:failures.append('compatibility_hash:'+rel)
    for o in ops:
        p=ROOT/o['path']
        if p.exists():
            present=list(files(p))
            if o['action']!='archive' or not present or any(q.relative_to(ROOT).as_posix() not in allowed for q in present):
                failures.append('source_not_removed:'+o['path'])
        if o['action']=='archive' and size(ARCHIVE/o['path'])!=o['bytes']:
            failures.append('archive_size:'+o['path'])
    out=dict(protected_artifacts=len(protected),operations=len(ops),failures=failures,
             current_bytes=size(ROOT),archive_bytes=size(ARCHIVE))
    write(AUDIT/'verification.json',out);print(json.dumps(out,ensure_ascii=True))
    if failures:sys.exit(1)


if __name__=='__main__':
    {'plan':make_plan,'execute':execute,'verify':verify}[sys.argv[1]]()

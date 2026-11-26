"""Restore the two dynamic import dependencies demonstrated by parity tests."""
from pathlib import Path
import shutil
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT.parent/'GAN对流层_历史归档_20261123'
rows=[]
for rel in ['research_pipeline/studies/europe_dense_zwd_20261104_v1/03_code/model/network.py',
            'research_pipeline/studies/europe_dense_zwd_20261104_v1/03_code/training/core.py']:
    source=ARCHIVE/rel;target=ROOT/rel
    assert source.is_file() and not source.is_symlink()
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert source.read_bytes()==target.read_bytes()
    else:shutil.copy2(source,target)
    h=hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    rows.append(dict(path=rel,sha256=h,reason='Dynamic us_gan import required by four actual model-parity tests'))
readme=ROOT/'research_pipeline/studies/europe_dense_zwd_20261104_v1/README.md'
readme.write_text('# 当前工程兼容源文件\n\n这里只保留模型一致性测试动态导入所需的 network.py 和 core.py。欧洲数据、训练结果及研究资料已归档，不属于当前研究区域。恢复记录见项目 maintenance/cleanup_20261123/retained_compatibility_files.json。\n',encoding='utf-8')
rows.append(dict(path=readme.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(readme.read_bytes()).hexdigest(),reason='Compatibility-only directory label'))
(ROOT/'maintenance/cleanup_20261123/retained_compatibility_files.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'restored_source_files':2,'identical_to_archive':True}))

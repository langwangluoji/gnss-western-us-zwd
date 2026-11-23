"""Generate current-region navigation after the approved filesystem cleanup."""
from pathlib import Path
import json
import csv
import hashlib

ROOT=Path(__file__).resolve().parents[2]
PIPE=ROOT/'research_pipeline'
STUDY=PIPE/'studies/us_dense_zwd_20261105_v1'
PROJECT=STUDY/'03_code/gnss_field_server'
DOCS=ROOT/'docs'
AUDIT=Path(__file__).resolve().parent


def main():
    rows=[]
    for p in sorted((PROJECT/'results').iterdir(),reverse=True):
        if not p.is_dir():continue
        candidates=[p/n for n in ['RESULTS.md','VERDICT.md','README.md','REPORT.md','progress.md','PLAN.md','comparison.json','results.json','status.json'] if (p/n).exists()]
        if not candidates:candidates=sorted(p.glob('*.md'))[:1]+sorted(p.glob('*.json'))[:1]
        status='见报告/锁文件'
        if (p/'status.json').exists():
            try:
                obj=json.loads((p/'status.json').read_text(encoding='utf-8'))
                if isinstance(obj,dict):status=str(obj.get('stage',obj.get('status',obj.get('phase',status))))
            except (ValueError,OSError):pass
        links='；'.join(f'[{q.name}](../{q.relative_to(ROOT).as_posix()})' for q in candidates[:3])
        rows.append(f'| {p.name} | {status.replace(chr(124), chr(47))[:100]} | {links} |')
    (DOCS/'EXPERIMENT_INDEX.md').write_text('# 当前区域实验索引\n\n以完整实验目录和锁文件定位。目录日期是实验标识，不代表评价数据年份；状态仅转录现有文件，不推断科学成功。最新主结论见 [CURRENT_RESEARCH.md](CURRENT_RESEARCH.md)。\n\n| 实验 | 已记录状态 | 证据入口 |\n|---|---|---|\n'+'\n'.join(rows)+'\n\n实际全生成验收另保留在工程 `build/full_generative_real_check_v01/v02/v03`，含独立权重与恢复检查；这些目录不属于普通打包暂存。初次扩区划分修正及暂停记录保留在研究目录 `04_models/field_generation_v1`。\n',encoding='utf-8')
    artifacts=[]
    for p in PROJECT.rglob('*'):
        if p.is_file() and p.suffix in {'.pt','.pth'} and '__pycache__' not in p.parts:
            artifacts.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size})
    protected=json.loads((AUDIT/'protected_artifacts_before.json').read_text(encoding='utf-8'))
    checksums={x['path']:x['sha256'] for x in protected}
    for a in artifacts:
        a['protected_before_cleanup']=a['path'] in checksums
        a['sha256']=checksums.get(a['path'])
        if not a['sha256']:
            h=hashlib.sha256()
            with (ROOT/a['path']).open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            a['sha256']=h.hexdigest()
    (AUDIT/'local_weights_manifest.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2),encoding='utf-8')
    (DOCS/'LOCAL_ARTIFACTS.md').write_text('''# 本地数据、权重与恢复

Git仓库同步源码、配置、测试、报告、科学锁、轻量指标和部分图件。原始观测、地形栅格、训练权重、大型预测数组和数据缓存保留在本机，克隆仓库不等于取得这些大文件。

## 数据位置

- `research_pipeline/data/pwv_data/`：当前美国西部边界内候选站的原始 `.pwv.gz`，保留原始全时段资料，不截断观测。分析数据仅按2016—2020协议提取。其他区域原始文件已外移归档。
- `research_pipeline/data/external/egm96_15.gtx` 与 `us_nga_egm96_15.tif`：高程基准。
- `research_pipeline/studies/us_dense_zwd_20261105_plan/evidence/countries.geojson`：当前陆地域定义的上游依赖。
- `research_pipeline/studies/us_dense_zwd_20261105_v1/02_data/raw/srtm/`：官方SRTM下载。
- 该研究的 `02_data/processed/field_generation_v1/`：站点、几何、角色及原始提取缓存。
- 该研究的 `02_data/processed/gnss_field_server_v1/`：2016—2020、A/B/C/D角色月片。`manifest.json`记录每个数组的形状和SHA256；Git保留清单，本地保留数组。

## 权重与预测

工程 `results/` 各实验内保存训练权重、选择锁、预测数组和指标。三个真实工程验收目录 `build/full_generative_real_check_v01/v02/v03` 原位保留。权重索引见 `maintenance/cleanup_20261123/local_weights_manifest.json`；本次整理保护文件的SHA256见 `protected_artifacts_before.json`。

迁移时应复制上述数据目录和所需实验目录，保持根目录相对结构，核验manifest和权重哈希。历史锁可能记录原Windows绝对路径；复制后先核对身份并显式登记新运行，不能静默改写旧锁。`configs/local.yaml`的数据相对位置保持可用。上游解析代码依赖h5py、rasterio、shapely等额外包；日常当前工程依赖由pyproject.toml定义。

保留的 `residual_edm_v1` 包提供当前需要的模型/工具支持；其 `early_mean.py` 等旧训练入口仍需从归档恢复旧数据与额外模块后才能运行。`studies/europe_dense_zwd_20261104_v1` 只保留一致性测试所需两个源文件，不保留欧洲实验进展。

## 已删除的可重建缓存

只删除了完整特征缓存及Python/pytest临时缓存。原始数据、来源权重、实验锁、预测和指标均保留。两组大缓存的原元数据保存在本机 `maintenance/cleanup_20261123/deleted_cache_metadata`。

1. `results/20261121_grid_processor_pair_v01/scientific/cache`：约31.60 GiB。由 `tools/grid_pair_data.py` 中 `GridStudy.prepare()` 及 `tools/grid_pair_train.py` 使用的入口重建。重建会重新计算冻结输入特征；先查看该实验锁和数据身份，不能把重建当成重新训练授权。
2. `results/20261122_local_generative_architecture_v01/cache`：约3.24 GiB。由同实验 `pilot_data.py` 的 `prepare()` 重建，依赖保留的来源数据和其他缓存。

缓存游标与数组一起删除，避免把空数组当成已完成缓存。调用旧实验脚本前必须先恢复所需缓存；已有指标的阅读和当前最新U-Net/校正不依赖这两个已删除缓存。

## 历史归档

同盘项目外 `../GAN对流层_历史归档_20261123/` 保存旧区域、旧模型、历史汇报、非当前区域原始数据及打包副本。归档按原项目相对路径组织；恢复前查看操作清单，不覆盖当前研究。
''',encoding='utf-8')
    # Preserve original readmes as provenance, replace stale entry points.
    readmes=[(PIPE/'README.md','../README.md'),(PIPE/'studies/README.md','../../README.md'),(STUDY/'README.md','../../../README.md')]
    for path,link in readmes:
        if path.exists():
            old=AUDIT/'previous_navigation'/path.relative_to(ROOT)
            old.parent.mkdir(parents=True,exist_ok=True)
            if not old.exists():old.write_bytes(path.read_bytes())
        path.write_text(f'# 当前研究入口\n\n美国西部25–50°N、125–110°W，2016—2020。请从[项目首页]({link})进入最新进展与实验索引。旧区域已移出项目；本目录保留当前工程、数据与上游依赖。\n',encoding='utf-8')
    readme=PROJECT/'README.md'
    content=readme.read_text(encoding='utf-8')
    intro='> 当前科学进展请先读 [项目进展](../../../../../docs/CURRENT_RESEARCH.md) 和 [实验索引](../../../../../docs/EXPERIMENT_INDEX.md)。下文为2026-11-18的工程与迁移说明，历史验收项及服务器待办以对应记录为准；不代表11月23日最新模型排名。构建/发行暂存副本已外移，需使用源码重新打包。\n\n'
    if not content.startswith('> 当前科学进展'):readme.write_text(intro+content,encoding='utf-8')
    print(json.dumps({'experiments':len(rows),'weight_files':len(artifacts)},ensure_ascii=True))


if __name__=='__main__':main()

"""Reports consume committed results; they never start training or select on 2020."""
from pathlib import Path
import json

from gnss_field.artifacts import atomic_bytes, read_json


def build_report(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob('metrics.json')):
        value = read_json(path)
        metrics = value.get('metrics', value)
        primary = metrics.get('primary', {})
        joint = metrics.get('joint', {})
        rows.append((path.relative_to(root).as_posix(), value.get('role', ''), value.get('members', ''),
                     primary.get('station_macro_rmse_mm'), primary.get('station_macro_crps_mm'),
                     joint.get('group_macro_es_mm'), joint.get('group_macro_vs_mm'),
                     primary.get('n_stations'), primary.get('n_station_hours')))
    lines = ['# GNSS场实验报告', '', '本报告读取已生成的评价产物；25 km表示输出间距。', '',
             '| 结果 | 角色 | K | 站宏RMSE/mm | CRPS/mm | ES/mm | VS | 站数 | 站时刻数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| '+' | '.join('—' if x is None else str(x) for x in row)+' |')
    if not rows:
        lines += ['', '尚无正式评价结果。工程测试中的损失和数值一致性不作为精度结论。']
    locks = list(root.rglob('product_lock.json'))
    lines += ['', f'发现{len(locks)}份产品锁定记录。点值、细尺度、边缘概率与联合概率收益分别评定。', '']
    status = root/'manifest.json'
    if status.exists():
        lines += ['```json', json.dumps(read_json(status), ensure_ascii=False, indent=2), '```', '']
    path = root/'REPORT.md'
    atomic_bytes(path, '\n'.join(lines).encode('utf-8'))
    return path

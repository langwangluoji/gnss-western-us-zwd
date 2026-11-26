from maintenance.paper_restructure_20261126.relocate_links import relocated_link


def test_report_relative_link_survives_directory_depth_change():
    mapping={'old/deep/run/report.md':'experiments/us/run/report.md',
             'old/deep/run/figures/a.png':'experiments/us/run/figures/a.png',
             'maintenance/europe/REPORT.md':'research/europe/REPORT.md'}
    assert relocated_link('old/deep/run/report.md','experiments/us/run/report.md','figures/a.png',mapping)=='figures/a.png'
    assert relocated_link('docs/current.md','docs/current.md','../maintenance/europe/REPORT.md#result',mapping)=='../research/europe/REPORT.md#result'
    assert relocated_link('a.md','archive/a.md','https://example.com/x',mapping)=='https://example.com/x'

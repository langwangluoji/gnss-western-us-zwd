from tools.terrain_diffusion.e01_e05_experiment import run

def test_contract_smoke(tmp_path):
    d=run("E01","random",5,tmp_path)
    assert d["station_contract"] == [410,137]
    assert d["forbidden"] == ["PWV","IWV","TCWV","C","D"]
    assert d["switches"]["terrain"] and not d["switches"]["edges"]

def test_e05_switches(tmp_path):
    d=run("E05","hole100",5,tmp_path)
    assert all(d["switches"].values())

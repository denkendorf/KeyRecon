import json
from importlib.resources import files

def test_en_reference_profile_is_packaged():
    path = files("keyrecon").joinpath("profiles/en_reference.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["profile"] == "en_reference"
    assert data["decision"]["round_decimals"] == 12
    assert data["decision"]["paper_reference_thresholds"]["Corpus B"] == 0.35

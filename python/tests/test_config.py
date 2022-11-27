from core.config import Config


def test_config_exists():
    assert Config.functionSet
    assert len(Config.dict) > 0

    for k, v in Config.dict.items():
        print(f"\t{k}: {v}")

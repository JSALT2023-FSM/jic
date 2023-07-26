from jic import datasets


def test_num_classes():
    assert len(datasets.SCENARIOS) == 18
    assert len(datasets.ACTIONS) == 54
    assert len(datasets.INTENTS) == 69


def test_valid_intents():
    for scenario, action in datasets.INTENTS:
        assert scenario in datasets.SCENARIOS
        assert action in datasets.ACTIONS

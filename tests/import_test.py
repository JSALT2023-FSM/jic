def test_import():
    import jic
    assert hasattr(jic, '__version__'), str(dir(jic))

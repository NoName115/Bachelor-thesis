from src.cnn_model.printer import print_info, print_warning, print_error

import pytest


def test_print_info():
    print_info('info')
    assert 1

def test_print_warning():
    print_warning('warning')
    assert 1

def test_print_error():
    with pytest.raises(SystemExit):
        print_error('error')

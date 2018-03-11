from ..cnn_model.printer import print_info, print_warning, print_error
from ..cnn_model.loader import DataLoader, DataSaver

import pytest

# -------------------
# ----- PRINTER -----
def test_print_info():
    print_info('info')
    assert 1

def test_print_warning():
    print_warning('warning')
    assert 1

def test_print_error():
    with pytest.raises(SystemExit):
        print_error('error')

# ------------------
# -- LOADER/SAVER --
def test_load_save_model():
    model_class, preproc = DataLoader.load_model_data(
        'src/tests/models/KerasBlog_2018-03-11_12:03:41/'
    )
    DataSaver.save_model('src/tests/models/', model_class, preproc)
    assert 1

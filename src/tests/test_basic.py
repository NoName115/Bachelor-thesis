import pytest
import sys
sys.path.insert(0, '/home/rkolcun/BP/src/cnn_model')

from printer import print_info, print_warning, print_error
from loader import DataLoader, DataSaver


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
        'src/tests/models/KerasBlog/'
    )
    DataSaver.save_model(
        'src/tests/models/',
        model_class, preproc
    )
    assert 1

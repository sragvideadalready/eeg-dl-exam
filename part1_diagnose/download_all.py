from concurrent.futures import ThreadPoolExecutor
from mne.datasets import eegbci

def get(s):
    eegbci.load_data(s, [6, 10, 14])
    return s

with ThreadPoolExecutor(5) as ex:
    for s in ex.map(get, [6, 7, 8, 9, 10]):
        print("done subject", s, flush=True)
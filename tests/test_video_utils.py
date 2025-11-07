import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.video_utils import get_frame_indices


def test_get_frame_indices_inclusive():
    random.seed(0)
    indices = [get_frame_indices(2, 4, sample='rand') for _ in range(100)]
    first_interval = [idx[0] for idx in indices]
    second_interval = [idx[1] for idx in indices]
    assert 1 in first_interval and 3 in second_interval

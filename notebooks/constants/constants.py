import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

LOCAL_DATA = os.path.join(PROJECT_ROOT, 'data')

BUCKET_NAME = 'sar-dataset'

DATA_ROOT = os.path.join(BUCKET_NAME, 'data')

TRAIN_DATA = 'data/new_dataset3/train'

VAL_DATA = 'data/new_dataset3/val'

TEST_DATA = 'data/new_dataset3/test'

ALL_LABELS = 'data/new_dataset3/All labels with Pose information/labels'
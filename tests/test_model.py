import numpy as np

from src import config
from src.models.cnn import build_cnn


def test_output_shape_and_softmax():
    model = build_cnn(use_augmentation=True)
    out = model.predict(np.random.rand(2, *config.INPUT_SHAPE).astype("float32") * 255, verbose=0)
    assert out.shape == (2, config.NUM_CLASSES)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_augmentation_inactive_at_inference():
    model = build_cnn(use_augmentation=True)
    x = np.random.rand(1, *config.INPUT_SHAPE).astype("float32") * 255
    assert np.allclose(model.predict(x, verbose=0), model.predict(x, verbose=0))

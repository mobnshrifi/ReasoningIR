import torch

from reasoning_ir import NAFNet, RecurrentNAFNet


def test_nafnet_shape():
    model = NAFNet()
    inp = torch.randn(2, 3, 64, 64)
    out = model(inp)
    assert out.shape == inp.shape


def test_recurrent_iterations():
    model = RecurrentNAFNet()
    inp = torch.randn(1, 3, 32, 32)
    final, outputs = model(inp, num_iterations=3, return_sequence=True)
    assert final.shape == inp.shape
    assert len(outputs) == 3
    for tensor in outputs:
        assert tensor.shape == inp.shape

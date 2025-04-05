import pytest
import torch
import yaml
from model import PINN

# Load config for tests
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Fixture to initialize the model
@pytest.fixture
def pinn_model():
    return PINN(config)

def test_model_init(pinn_model):
    """Test PINN model initialization."""
    assert pinn_model.N_input == config["model"]["n_input"]
    assert pinn_model.N_output == config["model"]["n_output"]
    assert pinn_model.N_nodes == config["model"]["n_nodes"]
    assert pinn_model.N_layers == config["model"]["n_layers"]
    assert pinn_model.lr == config["model"]["lr"]
    assert pinn_model.param_1 == config["model"]["param_1"]
    assert pinn_model.param_2 == config["model"]["param_2"]
    assert pinn_model.param_3 == config["model"]["param_3"]

def test_forward_pass(pinn_model):
    """Test forward pass with a sample input."""
    x = torch.tensor([[0.5]], dtype=torch.float32)
    output = pinn_model.forward(x)
    assert output.shape == (1, config["model"]["n_output"])
    assert torch.is_tensor(output)

def test_predict(pinn_model):
    """Test predict method."""
    x = torch.tensor([[0.5]], dtype=torch.float32)
    prediction = pinn_model.predict(x)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, config["model"]["n_output"])

def test_invalid_config():
    """Test model initialization with invalid config."""
    invalid_config = {"model": {"n_input": -1}}  # Invalid input size
    with pytest.raises(Exception):
        PINN(invalid_config)
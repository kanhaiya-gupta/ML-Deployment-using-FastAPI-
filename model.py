import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PINN(nn.Module):
    def __init__(self, config):
        super(PINN, self).__init__()
        try:
            self.config = config
            self.N_input = config["model"]["n_input"]
            self.N_output = config["model"]["n_output"]
            self.N_nodes = config["model"]["n_nodes"]
            self.N_layers = config["model"]["n_layers"]
            self.lr = config["model"]["lr"]
            self.param_1 = config["model"]["param_1"]
            self.param_2 = config["model"]["param_2"]
            self.param_3 = config["model"]["param_3"]
            self.lambda1 = config["model"]["lambda1"]
            self.lambda2 = config["model"]["lambda2"]

            self.fcs = nn.Sequential(
                nn.Linear(self.N_input, self.N_nodes),
                nn.Tanh()
            )
            self.fch = nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(self.N_nodes, self.N_nodes),
                    nn.Tanh()
                ) for _ in range(self.N_layers - 1)
            ])
            self.fce = nn.Linear(self.N_nodes, self.N_output)
            logger.info("PINN model initialized successfully.")
        except KeyError as e:
            logger.error(f"Missing key in config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing PINN model: {e}")
            raise

    def forward(self, x):
        try:
            x = self.fcs(x)
            x = self.fch(x)
            x = self.fce(x)
            return x
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def predict(self, X):
        try:
            self.eval()
            with torch.no_grad():
                out = self.forward(X)
            logger.debug("Prediction made successfully.")
            return out.cpu().numpy()
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
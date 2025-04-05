import yaml
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from model import PINN
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error("config.yaml file not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config.yaml: {e}")
        raise

def load_data(config):
    try:
        data = pd.read_csv(config["data"]["oscillator_data"])
        data_soln = pd.read_csv(config["data"]["oscillator_soln"])
        X_data = torch.tensor(data["X_values"], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        Y_data = torch.tensor(data["Y_values"], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        X_data_eqn = torch.tensor(data_soln["X_values"], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        Y_data_eqn = torch.tensor(data_soln["Y_values"], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        logger.info("Data loaded and processed successfully.")
        return X_data, Y_data, X_data_eqn, Y_data_eqn
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(config, X_data, Y_data, X_data_eqn, Y_data_eqn):
    try:
        model = PINN(config).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"])
        epochs = config["model"]["epochs"]

        with mlflow.start_run() as run:  # Capture the run object
            run_id = run.info.run_id  # Get the run_id
            mlflow.log_params(config["model"])
            losses = []
            x_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True).to(DEVICE)
            x_physics = torch.linspace(0., 1., 30).view(-1, 1).requires_grad_(True).to(DEVICE)

            for ep in range(epochs):
                optimizer.zero_grad()
                nn_outputs = model(X_data)
                loss_data = torch.mean((nn_outputs - Y_data) ** 2)

                y_boundary = model(x_boundary)
                loss_bound_a = (torch.squeeze(y_boundary) - 1) ** 2
                dydt = torch.autograd.grad(y_boundary, x_boundary, torch.ones_like(y_boundary), create_graph=True)[0]
                loss_bound_b = (torch.squeeze(dydt) - 0) ** 2

                y_physics = model(x_physics)
                dydx = torch.autograd.grad(y_physics, x_physics, torch.ones_like(y_physics), create_graph=True)[0]
                d2ydx2 = torch.autograd.grad(dydx, x_physics, torch.ones_like(dydx), create_graph=True)[0]
                physics_eqn = model.param_1 * d2ydx2 + model.param_2 * dydx + model.param_3 * y_physics
                loss_physics = torch.mean(physics_eqn ** 2)

                loss = loss_data + model.lambda2 * loss_physics
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if ep % (epochs // 10) == 0:
                    logger.info(f"Epoch {ep}/{epochs}, loss: {losses[-1]:.2f}")
                    mlflow.log_metric("loss", losses[-1], step=ep)

            # Define an input example and convert to NumPy array
            input_example = torch.tensor([[0.5]], dtype=torch.float32).to(DEVICE)
            input_example_numpy = input_example.cpu().numpy()  # Convert to NumPy

            # Log the model with the NumPy input example
            mlflow.pytorch.log_model(
                model,
                "pinn_model",
                input_example=input_example_numpy
            )
            logger.info(f"Model training completed. Run ID: {run_id}")

            # Save run_id to a file
            with open("run_id.txt", "w") as f:
                f.write(run_id)
            logger.info(f"Run ID saved to run_id.txt: {run_id}")

            # Plotting
            plt.plot(losses)
            plt.grid()
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.savefig("loss_plot.png")
            mlflow.log_artifact("loss_plot.png")

            Y_data_preds = model.predict(X_data_eqn)
            plt.figure()
            plt.plot(X_data_eqn.cpu()[:, 0], Y_data_eqn.cpu()[:, 0], alpha=0.8, label="Equation")
            plt.plot(X_data.cpu()[:, 0], Y_data.cpu()[:, 0], 'o', label="Training Data")
            plt.plot(X_data_eqn.cpu()[:, 0], Y_data_preds[:, 0], alpha=0.8, label="PINN")
            plt.legend()
            plt.grid()
            plt.ylabel('Y')
            plt.xlabel('t')
            plt.savefig("prediction_plot.png")
            mlflow.log_artifact("prediction_plot.png")
            logger.info("Plots generated and logged as artifacts.")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
    DEVICE = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["training"]["seed"])
    X_data, Y_data, X_data_eqn, Y_data_eqn = load_data(config)
    train_model(config, X_data, Y_data, X_data_eqn, Y_data_eqn)
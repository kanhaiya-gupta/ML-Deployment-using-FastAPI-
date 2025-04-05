# ML Deployment using FastAPI

This repository contains a project for deploying Machine Learning models using FastAPI. FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Deploying Machine Learning models can be a challenging task, but with FastAPI, you can create a robust and high-performance API for your models with minimal effort. This project demonstrates how to deploy a simple ML model using FastAPI and provides a template for deploying your own models.

## Installation

To get started with this project, you need to clone the repository and install the required dependencies. Follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/kanhaiya-gupta/ML-Deployment-using-FastAPI-.git
    cd ML-Deployment-using-FastAPI-
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the FastAPI server, use the following command:
```bash
uvicorn main:app --reload
```
This will start the server at `http://127.0.0.1:8000`. You can access the API documentation at `http://127.0.0.1:8000/docs`.

## Project Structure

The project structure is as follows:
```
ML-Deployment-using-FastAPI-
│
├── main.py                 # The main entry point of the application
├── models                  # Directory to store your ML models
│   └── model.pkl
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── ...
```

## API Endpoints

The following endpoints are available in the API:

- `GET /`: Returns a welcome message.
- `POST /predict`: Accepts input data and returns the model prediction.

### Example Request

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_data": [...]
}'
```

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

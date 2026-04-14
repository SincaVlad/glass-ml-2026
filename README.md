# Glass ML 2026

A machine learning project leveraging chemical engineering principles to predict glass properties based on their raw elemental compositions. 

## The Project
This project applies machine learning to material science. By analyzing the oxide formulations and chemical makeup of various glasses, the model predicts key thermal properties (e.g., transition or melting temperatures) to streamline material design and testing. 

## The Dataset
The models are trained on data derived from the **SciGlass database**, utilizing a robust, cleaned dataset of approximately **29,000 unique glass compositions**.

## Results
The current predictive model achieves high accuracy and reliability, yielding the following metrics on the test set:
* **R² Score:** 0.95
* **RMSE:** 32.5 °C

## Roadmap
The next phase of this project is transitioning the core Python model into a fully deployable web application:
* **FastAPI Backend:** To serve the machine learning model and handle inference requests.
* **Streamlit Frontend:** To provide an interactive, user-friendly interface where users can input custom glass compositions and receive real-time predictions.

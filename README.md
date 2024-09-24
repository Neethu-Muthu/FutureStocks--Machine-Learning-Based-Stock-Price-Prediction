# 📈 Stock Price Prediction using LSTM, GRU, and SVR

![ML Models](https://img.shields.io/badge/ML%20Models-LSTM%20%7C%20GRU%20%7C%20SVR-blue?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12345678/stock_market.png" alt="Stock Prediction" width="400px"/>
</p>

---

## 🚀 Project Overview

The **Stock Price Prediction** project utilizes advanced machine learning models — **LSTM (Long Short-Term Memory)**, **GRU (Gated Recurrent Units)**, and **SVR (Support Vector Regression)** — to predict future stock prices based on historical stock market data. The goal is to provide investors and traders with reliable predictions to make informed decisions.

**Why stock price prediction?**

- 📊 **Informed Decisions**: Help investors predict future stock values to make smarter investments.
- 💼 **Business Growth**: Aid businesses in financial forecasting and planning.
- ⚙️ **Automation**: Automate the analysis of historical data for stock market forecasting.

---

## 📦 Features

- 🧠 **Machine Learning Models**: LSTM, GRU, and SVR for effective time-series forecasting.
- 📈 **Historical Data Analysis**: Train models using historical stock data like Open, High, Low, Close prices.
- 🧮 **Evaluation Metrics**: Assess model performance using MSE, MAE, R² Score, and more.
- 📊 **Visualizations**: Interactive graphs comparing actual vs predicted stock prices.

---

## 🧩 Machine Learning Models Used

| Model | Description |
| ----- | ----------- |
| 🧠 **LSTM** | Long Short-Term Memory model captures long-term dependencies in sequential data. |
| 🧠 **GRU**  | Gated Recurrent Units optimize performance by simplifying LSTM architecture. |
| 📊 **SVR**  | Support Vector Regression fits the best hyperplane for stock price prediction. |

---

## 📑 Prerequisites

Before running this project, ensure the following libraries are installed:

- **Python 3.8+**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **TensorFlow 2.x**
- **Plotly**

Use the following command to install dependencies:

```bash
pip install -r requirements.txt

## 🛠️ Steps Involved

1. **📊 Data Preprocessing**: Load the historical stock price data and scale it to prepare for model training.
2. **🤖 Model Training**: Train LSTM, GRU, and SVR models on the processed stock data.
3. **📈 Prediction**: Use the trained models to predict future stock prices based on test data.
4. **⚖️ Model Evaluation**: Evaluate and compare the accuracy of predictions using metrics like MSE, MAE, and R² score.
5. **📊 Visualization**: Display the predicted stock prices alongside the actual prices for comparison.

---

## 🏆 Results

The stock price prediction results are evaluated using various metrics, allowing comparison between the models. The project outputs **visualizations** showing how closely the predicted prices match the actual historical prices, helping to assess the reliability of each model.

---

## 📊 Visualizations

The project uses **Plotly** to create interactive visualizations that:
- 📉 Compare predicted and actual stock prices.
- 📆 Show stock price trends over time.
- 📊 Highlight model performance through charts.

---

## 🔮 Future Enhancements

- **⚙️ Model Tuning**: Experiment with hyperparameter optimization to further improve model accuracy.
- **⏲️ Real-Time Data Integration**: Implement real-time stock price prediction using live data feeds.
- **🌐 Incorporating External Factors**: Integrate external variables such as news sentiment or economic indicators to refine predictions.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


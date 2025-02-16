# Temperature Prediction Using LSTM with TFLite

This project implements a **Long Short-Term Memory (LSTM) model** using TensorFlow/Keras to predict temperature **1 hour into the future**. The trained model is converted to **TensorFlow Lite (TFLite)** for deployment on edge devices.

## **Project Overview**
- The dataset consists of hourly temperature and humidity readings.
- The data is preprocessed, normalized, and split into training, validation, and test sets.
- A **LSTM model** is trained on the dataset.
- The trained model is converted into **TFLite format** for efficient deployment.
- The model is profiled for compatibility with **Edge Impulse** for ESP32 deployment.

## **Dataset**
The dataset (`Lawrence_Data.csv`) contains **961** rows with:
- **Datetime** (`datetime`): Timestamp of the recorded data.
- **Temperature** (`temp`): Temperature in degrees Celsius.
- **Humidity** (`humidity`): Relative humidity in percentage.

### **Data Preprocessing**
1. **Splitting Data**:
   - **Training Data**: 768 rows (80% of data).
   - **Validation Data**: 96 rows.
   - **Test Data**: 97 rows.

2. **Normalization**:
   - Data is **normalized** using `MinMaxScaler` between `0` and `1`.
   - The scaler parameters (`scale, min, data_min, data_max`) are saved in `scaler_params.txt` for later use.

3. **Feature Engineering**:
   - **Input Features**: `temp` and `humidity`
   - **Target Variable**: `temp` (temperature prediction)
   - **Sequence Length**: **24 hours** (1 day of data for prediction)

---

## **LSTM Model**
A **LSTM-based neural network** is built with:
- **Input Shape**: `(24, 2)` → 24 hours of `temp` and `humidity`.
- **Architecture**:
  - **LSTM Layer**: 128 units.
  - **Dense Layer**: 1 neuron (for temperature prediction).
- **Loss Function**: Mean Squared Error (`MSE`).
- **Optimizer**: Adam.
- **Training**: Runs for **20 epochs** with **early stopping**.

### **Model Training**
The model is trained with:
```python
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

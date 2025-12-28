# Edge AI Smart Helmet for Industrial Hazard Detection

## 📌 Project Overview
The **Edge AI Smart Helmet for Industrial Hazard Detection** is an intelligent wearable safety system designed to enhance worker protection in hazardous industrial environments such as construction sites, mines, chemical plants, and manufacturing units.

The system integrates **multi-sensor data fusion** with **on-device machine learning (Edge AI)** to continuously monitor environmental and physiological parameters and to classify worker safety status in real time as **Safe** or **Unsafe**, without relying on cloud connectivity.

---

## 🎯 Objectives
- Improve industrial worker safety using real-time monitoring
- Detect hazardous environmental conditions at the edge
- Reduce response time by avoiding cloud dependency
- Deploy lightweight ML models on resource-constrained hardware
- Enable predictive and proactive accident prevention

---

## 🧠 System Architecture
The smart helmet is built around an **ESP32 microcontroller** and follows a dual-path processing pipeline:

1. **Multi-Sensor Data Acquisition**
2. **Preprocessing & Feature Extraction**
3. **Machine Learning Inference (Edge AI)**
4. **Hazard Classification (Safe / Unsafe)**
5. **Alert & Notification System**

All inference is performed **locally on the device**, ensuring low latency and high reliability.

---

## ⚙️ Hardware Components
- ESP32 Microcontroller  
- MPU6050 (Accelerometer & Gyroscope)  
- MQ-2 Gas Sensor  
- Flame Sensor  
- DHT11 Temperature & Humidity Sensor  
- IR Sensor (Helmet Wear Detection)  
- Buzzer & LED Indicators  
- Rechargeable Battery  

---

## 🧪 Sensor Parameters Monitored
- Ambient temperature and humidity  
- Toxic and combustible gas concentration  
- Fire and flame presence  
- Worker motion and fall detection  
- Helmet usage compliance  
- Physiological activity (pulse patterns)

---

## 🤖 Machine Learning Models Used
The system evaluates multiple ML models for safety classification:

- **Support Vector Machine (SVM)**  
- **Artificial Neural Network (ANN)**  
- **1D Convolutional Neural Network (CNN)**  

### Model Highlights
- SVM achieved highest recall for safety-critical detection  
- CNN offered lowest inference latency after quantization  
- ANN provided a balance between accuracy and computation  

All models were **quantized and deployed on ESP32** using Edge AI techniques.

---

## 📊 Performance Summary
- Accuracy: **>99.5% across all models**
- Real-time inference on embedded hardware
- Low power consumption
- Reliable offline operation

---

## 🚨 Alert & Safety Mechanism
- Visual alerts using LEDs
- Audible alerts via buzzer
- Time-stamped hazard indication
- Immediate unsafe-condition detection

---

## 🏭 Applications
- Construction sites  
- Mining and underground operations  
- Oil & gas industries  
- Chemical and thermal plants  
- Smart factories (Industry 4.0)  

---

## 🚀 Key Features
- Edge AI–based real-time hazard detection  
- Multi-sensor data fusion  
- Internet-independent operation  
- Low latency and low power design  
- Scalable and wearable architecture  

---



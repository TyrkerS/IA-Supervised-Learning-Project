# SUPERVISED LEARNING PROJECT - CAR TYPE CLASSIFICATION

**Language:** Python  
**Environment:** Jupyter Notebook  
**README Language:** English

---

## ⭐ Project Summary
This project implements a complete supervised learning workflow using **PyTorch**, following the structure of an academic practical assignment.  
The goal is to classify cars into **different car types** based on numerical and categorical features provided in the dataset `CarsData.csv`.

The notebook includes:
- Full exploratory data analysis (EDA)
- Data preprocessing
- Dataset and dataloader implementation
- Neural network model design (MLP)
- Training loop
- Evaluation (accuracy, confusion matrix, training curves)
- Experiments comparing multiple solutions and improvements

This project demonstrates the foundations of the deep‑learning workflow in a structured and reproducible way.

---

## 🧩 Technologies & Skills Demonstrated

### **Machine Learning / Deep Learning**
- Multi-Layer Perceptron (MLP) model
- Training loops with PyTorch
- Loss computation (CrossEntropyLoss)
- Optimizers (SGD / Adam)
- Evaluation metrics (accuracy, confusion matrix)
- Experiment design and comparison

### **Data Processing**
- Exploratory data analysis with Pandas
- Feature selection & preprocessing
- Data normalization
- Dataset → DataLoader pipeline
- Splits for train/validation/test

### **Software Engineering & Notebook Structure**
- Teacher-provided base code reused in Student sections
- Modular notebook sections
- Clean experimental workflow
- GPU/CPU device management

---

## 📁 Project Structure (Notebook Sections)

```
P3a.ipynb
│
├── Teacher: Initialization
│   ├── Imports
│   ├── Device selection (CPU/GPU)
│
├── Teacher: Base Code
│   ├── Data loading helpers
│   ├── Exploratory analysis helpers
│   ├── PyTorch Dataset class
│   ├── Model creation function
│   ├── Training function
│   ├── Evaluation functions
│
├── Student: Exploratory Data Analysis
│   ├── Feature-by-feature analysis
│   ├── Distributions and relationships
│   ├── Justifications and insights
│
├── Student: Experiments
│   ├── Baseline solution
│   ├── Improved preprocessing
│   ├── Improved MLP architectures
│   ├── Training & evaluation for each model
│
└── Student: Results
    ├── Accuracy
    ├── Confusion matrix
    ├── Training curves
    ├── Comparison between solutions
```

### Design Philosophy
- **Teacher sections** provide reusable utilities and ensure consistent structure.  
- **Student sections** require analytical thinking and experimentation.  
- Each experiment modifies only one component (preprocess or model) for fair comparison.

---

## 🔍 Project Details

### **1. Exploratory Data Analysis (EDA)**
The EDA includes:
- Distribution plots for each feature  
- Relationships between pairs of variables  
- Written reasoning for each characteristic  
- Selection of relevant features for classification  

This step ensures proper understanding of the dataset before modeling.

---

### **2. Data Preprocessing**
Performed with Pandas:
- Cleaning and formatting columns  
- Normalization or standardization  
- Selection of informative features  
- Train/validation/test split  

The preprocessed dataset is transformed into a PyTorch `Dataset`.

---

### **3. Neural Network Model (MLP)**
Core elements:
- Several `nn.Linear` layers  
- ReLU activations  
- Softmax output via `CrossEntropyLoss`  
- Configurable depth and width  

Improvements experimented with:
- Deeper architecture  
- More hidden units  
- Dropout or normalization (if implemented)  
- Optimizer variations (SGD vs Adam)

---

### **4. Training**
Implements:
- Epoch loop  
- Forward + backward pass  
- Loss monitoring  
- Accuracy tracking  
- GPU acceleration (if available)

Training time is reported for each solution.

---

### **5. Evaluation**
Includes:
- Final test accuracy  
- Confusion matrix  
- Analysis of misclassifications  
- Comparison between solutions  
- Discussion about performance differences  

---

## ▶️ How to Run the Project

### **1. Install dependencies**
```
pip install torch pandas numpy matplotlib scikit-learn
```

### **2. Open the notebook**
```
jupyter notebook P3.ipynb
```

### **3. Ensure dataset is available**
Place `CarsData.csv` in the same folder as the notebook.

### **4. Run the notebook sequentially**
- First run all **Teacher** cells  
- Then complete/execute the **Student** sections  
- Train all models  
- Compare results  

GPU will be used automatically if available.

---

## ✔ Summary
This project is a complete example of a supervised deep learning workflow using PyTorch, covering:
- EDA  
- Preprocessing  
- Model design  
- Training  
- Evaluation  
- Experimental comparison  

The final goal is to classify car types based on structured tabular data.  
It is an ideal introduction to applied machine learning with neural networks.


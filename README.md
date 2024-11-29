# **Intelligent Expense Tracker**

The **Intelligent Expense Tracker** is a Python-based project designed to help users classify their expenses using **machine learning**. With the help of a **Random Forest Classifier**, this project enables users to analyze their bank statements, categorize expenses, and gain insightful information through **Exploratory Data Analysis (EDA)**.

---

## **Features**
- **Expense Categorization**:
  - Leverage a pre-trained Random Forest Classifier to classify expenses into user-defined categories.
- **Customizable Training**:
  - Train the machine learning model using annotated CSV files for personalized categorization.
- **Bank Statement Integration**:
  - Parse bank statement PDFs to create training datasets automatically.
- **Exploratory Data Analysis (EDA)**:
  - Analyze categorized expenses to uncover patterns and trends in spending behavior.

---

## **Workflow**
1. **Input Bank Statements**:
   - Upload your bank statement PDFs.
   - The project extracts relevant information and creates structured CSV files.
2. **Annotate Data** (Optional):
   - Annotate the CSV files to categorize expenses as per your needs.
3. **Train the Model**:
   - Use annotated CSV files to train the Random Forest Classifier for expense categorization.
4. **Track Expenses**:
   - Input new bank statement data to classify expenses and generate detailed reports.
5. **Explore Insights**:
   - Perform EDA on categorized data to visualize spending habits and identify areas for improvement.

---

## **Technologies Used**
- **Programming Language**: Python
- **Machine Learning Algorithm**: Random Forest Classifier
- **Libraries and Tools**:
  - Data Processing: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`
  - PDF Parsing: `PyPDF2` or similar
  - Data Visualization: `matplotlib`, `seaborn`

---

## **Getting Started**

### **Prerequisites**
- Python 3.8 or above
- Install required libraries:
  ```bash
  pip install -r requirements.txt

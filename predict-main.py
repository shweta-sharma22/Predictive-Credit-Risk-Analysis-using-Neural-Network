import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
from streamlit_lottie import st_lottie

def preprocess_data(data):
    # Display original column names
    st.write("### Original Column Names:")
    st.write(data.columns.tolist())
    
    # Convert column names to title case
    data.columns = data.columns.str.title()
    
    # Display updated column names
    st.write("### Updated Column Names:")
    st.write(data.columns.tolist())
    
    # Check for null values in the dataset
    st.write("### Handling Null Values")
    st.write("Checking for null values in the dataset.")
    null_values = data.isnull().sum()
    if null_values.sum() > 0:
        st.warning("Warning: Null values found in the dataset.")
        st.write("Null Values:")
        st.write(null_values)
        st.write("Dropping rows with null values.")
        data.dropna(inplace=True)
        st.write("Null values have been dropped.")
    else:
        st.write("No null values found in the dataset.")
    
    non_numerical_columns = data.select_dtypes(exclude=['number']).columns
    categorical_cols = [col for col in non_numerical_columns if data[col].nunique() < len(data)]

    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    one_hot_encoded_data = pd.get_dummies(data, columns=categorical_cols, dtype=int)

    target_column = None
    if 'Target' in one_hot_encoded_data.columns:
        target_column = 'Target'
    elif 'Default' in one_hot_encoded_data.columns:
        target_column = 'Default'

    if target_column is None:
        raise ValueError("Neither 'Target' nor 'Default' column found in the dataset.")

    X = one_hot_encoded_data.drop(columns=[target_column])
    y = one_hot_encoded_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, classification_rep

def main():
    if "page" not in st.session_state:
        st.session_state.page = "intro"

    if st.session_state.page == "intro":
        path = r"C:\Users\shwet\OneDrive\Desktop\PROJECT\animation.json"
        with open(path, "r") as file:
            url = json.load(file)

        col1, col2 = st.columns([3, 3])

        with col1:
            st.title("Predictive Credit Risk Analysis")
            st.write("Our Credit Risk Predictor is a handy tool that helps you assess the risk associated with lending money. Just upload your dataset, and we'll crunch the numbers for you. Our tool provides easy-to-understand metrics like accuracy, confusion matrix, and classification report, giving you valuable insights into credit risk")

        with col2:
            st.lottie(url,
                      reverse=True,
                      height=400,
                      width=400,
                      speed=1,
                      loop=True,
                      quality='high',
                      key='lady'
                      )

        if st.button('Get started'):
            with st.spinner('Please wait...'):
                st.session_state.page = "main"

    elif st.session_state.page == "main":
        st.title("Predictive Credit Risk Analysis")
        if st.button('Back'):
            st.session_state.page = "intro"

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("### Preview of the uploaded dataset:")
            st.write(data.head())

            preprocess_button_placeholder = st.empty()
            preprocess_button = preprocess_button_placeholder.button('Preprocess Data')
            metrics_button_placeholder = st.empty()
            metrics_button = metrics_button_placeholder.button('Metrics')
            predict_button_placeholder = st.empty()
            predict_button = predict_button_placeholder.button('Predict')

            if preprocess_button:
                with st.spinner('Preprocessing Data...'):
                    try:
                        X_train, X_test, y_train, y_test = preprocess_data(data.copy())
                        st.success("Data has been preprocessed successfully!")
                    except Exception as e:
                        st.error("Error during data preprocessing:", e)

            if metrics_button:
                with st.spinner('Calculating Metrics...'):
                    try:
                        X_train, X_test, y_train, y_test = preprocess_data(data.copy())
                        y_pred = train_model(X_train, X_test, y_train, y_test)
                        accuracy, conf_matrix, classification_rep = evaluate_model(y_test, y_pred)

                        st.write("### Model Evaluation:")
                        st.write("### Accuracy")
                        st.write(accuracy)
                        
                        # Plot confusion matrix heatmap with explanation
                        st.write("### Confusion Matrix Heatmap:")
                        st.write("The confusion matrix heatmap below shows the relationship between actual and predicted labels. Each cell represents the number of data points that were predicted in a specific category. The brighter the color, the higher the number of predictions.")
                        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
                        plt.xlabel('Predicted labels')
                        plt.ylabel('True labels')
                        plt.title('Confusion Matrix')
                        st.pyplot(plt)

                        st.write("### Explanation:")
                        st.write("TP: True Positive (correctly predicted positive instances)")
                        st.write("FP: False Positive (incorrectly predicted positive instances)")
                        st.write("TN: True Negative (correctly predicted negative instances)")
                        st.write("FN: False Negative (incorrectly predicted negative instances)")

                        st.write("### Confusion Matrix:")
                        st.write(pd.DataFrame(conf_matrix, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive']))

                        # Explanation for classification report
                        st.write("### Classification Report:")
                        st.write("The classification report provides a summary of different evaluation metrics like precision, recall, F1-score, and support for each class in the dataset. It helps in understanding how well the model performs for each class.")

                        report_data = []
                        lines = classification_rep.split('\n')
                        for line in lines[2:-5]:
                            row = line.split()
                            report_data.append(row)
                        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
                        st.write(report_df)

                    except Exception as e:
                        st.error("Error during model evaluation:", e)

            if predict_button:
                with st.spinner('Predicting...'):
                    try:
                        X_train, X_test, y_train, y_test = preprocess_data(data.copy())
                        y_pred = train_model(X_train, X_test, y_train, y_test)

                        st.write("Predictions:")
                        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                        st.write(results)

                        # Visualization for comparison between actual and predicted values
                        st.write("### Comparison between Actual and Predicted Values:")
                        st.write("This visualization compares the actual labels with the predicted labels.")
                        results['Actual'].value_counts().plot(kind='bar', color='blue', alpha=0.5, label='Actual')
                        results['Predicted'].value_counts().plot(kind='bar', color='red', alpha=0.5, label='Predicted')
                        plt.xlabel('Labels')
                        plt.ylabel('Counts')
                        plt.title('Comparison')
                        plt.legend()
                        st.pyplot(plt)

                    except Exception as e:
                        st.error("Error during prediction:", e)

if __name__ == "__main__":
    main()
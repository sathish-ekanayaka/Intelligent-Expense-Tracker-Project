from src.required_classes import modelOperations,dataAnalytics,fileOperations,dataAnalytics
import streamlit as st
import os
import shutil

st.set_page_config(layout="wide")

st.title("Upload Your Monthly Bank Statement PDF file")

# Upload PDF file
uploaded_file = st.file_uploader("Upload Your Monthly Bank Statement PDF file", type=["pdf"])

if uploaded_file:
    # Display the file name
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Save the file to a temporary location
    temp_file_path = os.path.join("temp", uploaded_file.name)
    
    # Ensure the temp directory exists
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

    # Write the file to the temporary directory
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the file path
    st.write(f"File saved to: {temp_file_path}")


    dataframe = fileOperations()

    df_to_model = dataframe.pdf_file(temp_file_path)
    
    #Removing the file from the tempory file path to avoid disc clutter
    shutil.rmtree('temp')

    modelOps = modelOperations()

    output = modelOps.model_inference(df_to_model)

    analytics = dataAnalytics(output)
    print(output[['Date','Transaction_Category','Paid out']])

    fig1 = analytics.by_category()

    fig2 = analytics.plot_paid_out_by_day()
    
    col1, col2 = st.columns([3,3])

    # Display the plots
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)





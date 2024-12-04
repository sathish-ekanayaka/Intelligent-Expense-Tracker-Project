import numpy as np
import camelot
import PyPDF2
import pandas as pd
import re
import math
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timedelta
import pickle
import Levenshtein as lev
import os
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


###################################################################################
class fileUploads:
    
    
    def __init__(self):

        self.name = "Data Acquisision Class"

    def upload_file(self):
    # Create a hidden root window
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
        
        # Open the file dialog
            file_path = filedialog.askopenfilename(
                title="Select a file to upload",
                filetypes=[("All Files", "*.*"), ("Text Files", "*.txt"), ("CSV Files", "*.csv")]
            )
        
            if  file_path:
                print(f"Selected file: {file_path}")
                return str(file_path)
            else:
                print("No file selected.")
                return None
        
        except Exception as e:
            print("An Error Occured while uploading the file : {e}")
            return None

##################################################################################
class dataPreprocessing:
    
    def __init__(self):
        self.name = "Data preprocessing for model and inference"
 #Dataframe aquired from the PDF for training file preparation or inference is formatted into the desired format    
    def data_frame_arranging(self, df):

        def move_data(row):
            if '+' not in row['Paid in']:
                row['Balance'] = row['Paid in']
                row['Paid in'] =''
            return row
    
        df=df.apply(move_data,axis=1)
        df.reset_index(drop=True, inplace=True)

        def convert_to_date(text):
            try:
                return pd.to_datetime(text)
            except ValueError:
                return None

    # Apply the function to create a new column with datetime objects
        df['Date'] = df['Description'].apply(convert_to_date)
        df['Date'] = df['Date'].fillna(method='bfill')
        df.replace({' ': np.nan, '': np.nan}, inplace=True)
        df.dropna(subset=['Paid out'], how='all', inplace=True)
        df=df[['Date','Description','Paid out']]
        df.reset_index(drop=True, inplace=True)
        data2 = df
    #Adding a new data point for testing
        #data2.loc[116] = [pd.to_datetime('2024-01-29'),'VDC-CirkleK-Oil Top','-45.65']
    ########################################
        data2['Paid out'] = pd.to_numeric(data2['Paid out'].str.replace('-', '').str.strip(), errors='coerce')
        data2['Paid out']=pd.to_numeric(data2['Paid out'])
        data2['Paid out']=data2['Paid out'].abs()

        return data2

    # Cleaning Data for ML model (Training or Inference)
    def data_cleaning(self, df):
        
        df['Description'] = df['Description'].str.replace('VDC-', '').str.replace('VDP-', '')

        def text_cleaning(text):
            # Convert words to lower case.
            text = text.lower()
            # Remove special characters and numbers
            # which are not important in classifying expenses
            text = re.sub(r'[^\w\s]|https?://\S+|www\.\S+|https?:/\S+|[^\x00-\x7F]+|\d','', str(text).strip())

            # Tokenise
            text_list = word_tokenize(text)
            result = ''.join(text_list)
            return result
        
        df['Description'] = df['Description'].apply(lambda x: text_cleaning(x))
        return df
    
    #Creating attributes, preparing data to feed into the model.

    def data_refining(self, df):
        def separate_float(number):
            integer_part = math.floor(number)
            fractional_part = number - integer_part
            return integer_part, fractional_part
        
        # Counting the number of decimal places of a number
        def count_decimal_places(number):
            num_to_string = { 0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four'}
            return num_to_string[len(str(int(number)))]
        
        def is_there_fraction(number):
            if number==0.00:
                return "no"
            else:
                return "yes"
            
        df[['integer_part', 'fractional_part']] = df['Paid out'].apply(separate_float).apply(pd.Series)
        df['Decimal Places']=df['integer_part'].apply(count_decimal_places).apply(pd.Series)
        df=df.reset_index()
        df["Is there fraction"] = df['fractional_part'].apply(is_there_fraction).apply(pd.Series)

        return df

class fileOperations(fileUploads,dataPreprocessing):
    
    def __init__(self):
        super().__init__()
        self.name ="Data Preprocessing for training and inference"
    
    def pdf_file(self,pdf_path):
        
        try:
            #pdf_path = self.upload_file()
            tables = camelot.read_pdf(pdf_path,flavor='stream',pages="all")
            table1 =tables[0].df
            table2=tables[1].df
            data=[]
            for table in tables:
                data.append(table.df)
            df=pd.concat([data[0],data[1],data[2],data[3]]).fillna(0)
            df1=df.drop(df.index[0])
            df1.columns= df1.iloc[0]
            df1=df1.drop(df.index[1])
            df1=df1.fillna(0)
            df1=df1.astype(str)
            df1=self.data_frame_arranging(df1)
            return df1
        
        except FileNotFoundError as e:
            print(f"File Eroor : {e}")
        except ValueError as e:
            print(f"Value Error : {e}")
        except Exception as e:
            print(f"Unexpected Error occured while parsing the PDF : {e}")
            return None
    

    
    def csv_save_from_pdf(self):
        data_to_csv = self.pdf_file()
        data_to_csv['Transaction_Category'] = ""
        #folder_path = 'CSV_files_for_training'
        data_to_csv.to_csv('raw csvs/selected_columns.csv', index=False)
    
    
    
#################################################################################################
class modelOperations(fileOperations):
    
    def __init__(self):
        super().__init__()
        self.name = 'Model building, training and inference.'

    def model_training_initialization(self):
        
        csv_path = self.upload_file()
        try:
            train_data = pd.read_csv(csv_path)
        except FileNotFoundError:
            print("The training data file was not found. Please select a valid file.")
            return None
        except pd.errors.EmptyDataError:
            print("The selected file is empty. Please select a valid file.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while reading the CSV file: {e}")
            return None

        train_data=self.data_cleaning(train_data)
        train_data = self.data_refining(train_data)
        #Peparing the dictionary of transaction categories and descriptions
        transaction_chanels = train_data.groupby('Transaction_Chanel')['Description'].apply(lambda x: list(set(x))).to_dict()
        try:
            with open('pkl files/transaction_chanels_dict.pkl', 'wb') as file:
                pickle.dump(transaction_chanels, file)
        except FileNotFoundError:
            print("The transaction_model_dict file was not found. Please ensure the file exists in 'pkl files/'.")
            return None
        except pickle.UnpicklingError:
            print("Error unpickling the transaction_model_dict file. The file may be corrupted.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading the transaction_model_dict file: {e}")
            return None
            
        return train_data
        



    def model_training_process(self):
        data = self.model_training_initialization()
        used_columns = data[['Transaction_Chanel', 'Decimal Places', 'Is there fraction','Transaction_Category']]
        encoded_data = pd.get_dummies(used_columns, columns=['Transaction_Chanel', 'Decimal Places', 'Is there fraction'], drop_first=True)
        # Split data into features (X) and target (y)
        X = encoded_data.drop('Transaction_Category', axis=1)
        y = encoded_data['Transaction_Category']

        # Label encode the target 'Expense Category'
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        label_dict = {encoded: original for encoded, original in zip(y, data['Transaction_Category'])}

        
        with open('pkl files/label_dict.pkl', 'wb') as file:
            pickle.dump(label_dict, file)

        # Train-Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Model
        rf = RandomForestClassifier(random_state=7)
        rf.fit(X_train, y_train)
        with open('pkl files/random_forest_model.pkl', 'wb') as file:
            pickle.dump(rf, file)

        # Evaluate
        accuracy = rf.score(X_test, y_test)
        print(f'Random Forest Accuracy: {accuracy:.2f}')
        
        return y
                
    
    def model_inference(self,pdf_df):
        #initial_df = self.pdf_file()
        initial_df =self.data_cleaning(pdf_df)
        initial_df=self.data_refining(initial_df)
        
        def get_category(description):
            with open('pkl files/transaction_chanels_dict.pkl','rb') as file:
                category_dict = pickle.load(file)
            closest_match = None
            min_distance = float('inf')
            for category, items in category_dict.items():
                if description in items:
                    return category
        #Implementation of Lenshtein distance
            for key, values in category_dict.items():
                for value in values:
                    distance = lev.distance(description, value)
                    if distance < min_distance:
                        min_distance = distance
                        closest_match = (key, value)
            return key
        
        initial_df['Transaction_Chanel'] = initial_df['Description'].apply(get_category)

        df = initial_df[['Transaction_Chanel', 'Decimal Places', 'Is there fraction']]
        
        df = pd.get_dummies(df, columns=['Transaction_Chanel', 'Decimal Places', 'Is there fraction'], drop_first=True)
        
        try:
            with open('pkl files/random_forest_model.pkl', 'rb') as file:
                loaded_rf = pickle.load(file)
            targets = loaded_rf.predict(df)
        except FileNotFoundError:
            print("The model file was not found. Please ensure the file exists in 'pkl files/'.")
            return None
        except pickle.UnpicklingError:
            print("Error unpickling the model file. The file may be corrupted.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            return None

        try:
            with open('pkl files/label_dict.pkl','rb') as file:
                label_dict = pickle.load(file)
        except FileNotFoundError:
            print("The label_dict file was not found. Please ensure the file exists in 'pkl files/'.")
            return None
        except pickle.UnpicklingError:
            print("Error unpickling the label_dict file. The file may be corrupted.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while loading the label_dict: {e}")
            return None


        expense_category = pd.DataFrame({'Transaction_Category': targets})
        initial_df['Transaction_Category'] = expense_category['Transaction_Category'].map(label_dict)
    
        return initial_df
            


class dataAnalytics:
    def __init__(self, df_analytics):
        self.name = "Data Analytics Class"
        self.df_analytics = df_analytics

    def analytics_preperation(self):
        try:
    # Attempt to select the required columns
            df_analytics = self.df_analytics[['Date', 'Paid out', 'Transaction_Category']]
        except KeyError as e:
            # Handle missing columns
            missing_columns = set(['Date', 'Paid out', 'Transaction_Category']) - set(self.df_analytics.columns)
            raise KeyError(f"Missing required columns: {', '.join(missing_columns)}") from e
        except TypeError as e:
            # Handle invalid object type for self.df_analytics
            raise TypeError("self.df_analytics is not a valid DataFrame or subscriptable object.") from e
        except Exception as e:
            # Handle any unexpected errors
            raise RuntimeError(f"An unexpected error occurred: {str(e)}") from e        
        
        try:
            # Attempt to sort the DataFrame by the "Date" column
            df_analytics = df_analytics.sort_values(by="Date", ascending=True)
        except KeyError as e:
            # Handle missing "Date" column
            raise KeyError(f"The column 'Date' is missing in the DataFrame.") from e
        except TypeError as e:
            # Handle invalid or unsortable data in the "Date" column
            raise TypeError("The 'Date' column contains invalid or unsortable data (e.g., mixed types).") from e
        except AttributeError as e:
            # Handle invalid DataFrame object
            raise AttributeError("The object being sorted is not a valid DataFrame.") from e
        except Exception as e:
            # Handle any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred while sorting: {str(e)}") from e
                
        
        df_analytics.reset_index(drop=True, inplace=True)
        df_analytics["year_month"] = df_analytics["Date"].dt.to_period("M")
        # Count occurrences for each month
        month_counts = df_analytics["year_month"].value_counts()
        # Identify the month with the most data
        most_frequent_month = month_counts.idxmax()
        # Filter rows to keep only the data for the most frequent month
        df_analytics = df_analytics[df_analytics["year_month"] == most_frequent_month]
        # Drop the helper column (optional)
        df_analytics.drop(columns=["year_month"], inplace=True)
        #Adding a column saying which day of the week
        df_analytics["day_of_week"] = df_analytics["Date"].dt.day_name()
        return df_analytics
    
    def by_category(self):
        font = 40
        df_category = self.analytics_preperation()
        
        # Group by 'Transaction_Category' and calculate the sum of 'Paid out'
        category_sum = df_category.groupby('Transaction_Category')['Paid out'].sum()
        
        # Combine small categories into "Others" (if applicable)
        threshold = 0.05 * category_sum.sum()  # Categories contributing less than 5% of the total
        small_categories = category_sum[category_sum < threshold].sum()
        if small_categories > 0:
            category_sum = category_sum[category_sum >= threshold]
            category_sum['Others'] = small_categories
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Define colors using a modern palette
        colors = plt.cm.Paired(np.linspace(0, 1, len(category_sum)))
        
        # Plot pie chart with improved features
        wedges, texts, autotexts = ax.pie(
            category_sum,
            labels=category_sum.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'color': 'black', 'size' :30}
        )
        
        # Enhance text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(30)
        
        # Add a title
        ax.set_title('Expense Category Distribution (Paid Out)', pad=20, fontsize = 60)
        
        # Ensure the pie chart is a circle
        ax.axis('equal')
        
        # Return the figure
        return fig


    

    # Sample function to plot the data
    def plot_paid_out_by_day(self):
        font = 30
        df_day = self.analytics_preperation()
        
        # Group by 'day_of_week' and calculate the sum of 'Paid out'
        daily_sum = df_day.groupby('day_of_week')['Paid out'].sum().reset_index()
        
        # Sort by the chronological order of days (optional, if needed)
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        daily_sum['day_of_week'] = pd.Categorical(daily_sum['day_of_week'], categories=ordered_days, ordered=True)
        daily_sum = daily_sum.sort_values('day_of_week')
        
        # Create the plot
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(20,20))
        barplot = sns.barplot(
            data=daily_sum, 
            x='day_of_week', 
            y='Paid out', 
            palette='Blues_r', 
            ax=ax
        )
        
        # Annotate each bar with its value
        for bar in barplot.patches:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',  # Format the annotation
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # Offset text slightly above the bar
                textcoords='offset points',
                ha='center',
                va='center',
                fontsize = font,
                color='black'
            )
        
        # Add labels and title
        ax.set_title('Sum of Paid Out for Each Day of the Week', pad=20, fontsize = 60)
        ax.set_xlabel('Day of Week', labelpad=10, fontsize = font)
        ax.set_ylabel('Total Paid Out', labelpad=10, fontsize = font)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right',fontsize = font)
        ax.set_yticklabels(ax.get_yticklabels(), ha='right',fontsize = font)

        
        # Return the figure
        return fig



    


    
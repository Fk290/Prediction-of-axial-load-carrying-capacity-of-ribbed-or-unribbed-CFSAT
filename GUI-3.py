from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'

import pandas as pd
import numpy as np
from tkinter import messagebox

# Introduce database

dataset = pd.read_excel('compress dataset.xlsx')
dataset.head()
# print(dataset.head())

# Define the input and output characteristics
X = dataset.loc[:, dataset.columns != '抗压承载力(kN)']
y = dataset.loc[:, '抗压承载力(kN)']
# print(X.shape)
# print(y.shape)

# Create a StandardScaler object
scaler = StandardScaler()

# Use the fit method to calculate the mean and standard deviation of the data
scaler.fit(X)

# Using the transform method to normalize the database
X = scaler.transform(X)

# Dataset Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)

# Definition of XGBoost Model
model_1 = XGBRegressor(objective='reg:squarederror',random_state=15,min_child_weight=1,learning_rate=0.29927,n_estimators=498)
model_1.fit(X_train, y_train)


from joblib import dump, load
dump(model_1, 'XGBoost_model.joblib')
loaded_model = load('XGBoost_model.joblib')


# GUI interface settings

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# tkinter GUI
root = tk.Tk()
root.title("Prediction of axial load-carrying capacity of ribbed or unribbed CFSAT after high temperature")

canvas1 = tk.Canvas(root, width=950, height=750)
canvas1.configure(background='#e9ecef')
canvas1.pack()

# ... (rest of the GUI code)
# adding a label to the root window
label0 = tk.Label(root, text='Developed by Feng kai, Li Ming, Wanbo Yang', font=('Arial', 14, 'bold', 'italic'), bg='#e9ecef')
canvas1.create_window(20, 30, anchor="w", window=label0)
label_city = tk.Label(root, text='China, Suzhou University of Science and Technology ', font=('Arial', 14, 'bold', 'italic'), bg='#e9ecef')
canvas1.create_window(20, 62, anchor="w", window=label_city)
label_input = tk.Label(root, text='Input Parameters', font=('Arial', 14, 'bold', 'italic', 'underline'),
                       bg='#e9ecef')
canvas1.create_window(20, 100, anchor="w", window=label_input)

label1 = tk.Label(root, text='Thickness of aluminum alloy tube (mm) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 140, anchor="w", window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(460, 140, window=entry1,width=167)

label2 = tk.Label(root, text='Height of aluminum alloy tube (mm) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 170, anchor="w", window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(460, 170, window=entry2,width=167)

label3 = tk.Label(root, text='Diameter of aluminum alloy tube (mm) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 200, anchor="w", window=label3)

entry3 = tk.Entry(root)  # create 3rd entry box
canvas1.create_window(460, 200, window=entry3,width=167)

label4 = tk.Label(root, text='Width of rib (mm) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 230, anchor="w", window=label4)

entry4 = tk.Entry(root)  # create 4th entry box
canvas1.create_window(460, 230, window=entry4,width=167)

label5 = tk.Label(root, text='Length of rib (mm) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 260, anchor="w", window=label5)

entry5 = tk.Entry(root)  # create 5th entry box
canvas1.create_window(460, 260, window=entry5,width=167)

label6 = tk.Label(root, text='Number of rib :',font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 290, anchor="w", window=label6)

combo_box_1 = ttk.Combobox(root, values=["0", "3", "4", '5','6','8'])
canvas1.create_window(460, 290, window=combo_box_1,width=167)

label7 = tk.Label(root, text='Concrete compressive strength (Mpa) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 320, anchor="w", window=label7)

entry7 = tk.Entry(root)  # create 7th entry box
canvas1.create_window(460, 320, window=entry7, width=167)

label8 = tk.Label(root, text='Temperature (°C) :', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 350, anchor="w", window=label8)

entry8 = tk.Entry(root)
canvas1.create_window(460, 350, window=entry8,width=167)

label0000 = tk.Label(root, text='Output', font=('Arial', 14, 'bold', 'italic', 'underline'), bg='#e9ecef')
canvas1.create_window(20, 390, anchor="w", window=label0000)

label_out = tk.Label(root, text='Axial Load-Carrying Capacity (MPa)', font=('Arial', 14), bg='#e9ecef')
canvas1.create_window(20, 430, anchor="w", window=label_out)

text_label_1 = tk.Label(root, text="Range of Model Parameters",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 100, anchor="w", window=text_label_1)

text_label_2 = tk.Label(root, text="2.5 ≤ t ≤ 5",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 140, anchor="w", window=text_label_2)

text_label_3 = tk.Label(root, text="360 ≤ H ≤ 480",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 170, anchor="w", window=text_label_3)

text_label_4 = tk.Label(root, text="120 ≤ D ≤ 160",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 200, anchor="w", window=text_label_4)

text_label_5 = tk.Label(root, text="w=0 or 2.5 ≤ w ≤ 5",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 230, anchor="w", window=text_label_5)

text_label_6 = tk.Label(root, text="l=0 or 12 ≤ l ≤ 16",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 260, anchor="w", window=text_label_6)

text_label_7 = tk.Label(root, text="Choose from 0, 3, 4, 5, 6 or 8",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 290, anchor="w", window=text_label_7)

text_label_8 = tk.Label(root, text="30 ≤ fcu ≤ 70",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 320, anchor="w", window=text_label_8)

text_label_9 = tk.Label(root, text="20 ≤ T ≤ 400",font=('Arial', 14, 'italic'), bg='#e9ecef')
canvas1.create_window(600, 350, anchor="w", window=text_label_9)

# Loading image
label100 = tk.Label(root, text='Feature Importance Analysis Based on SHAP Method', font=('Arial', 14,  'underline'), bg='#e9ecef')
canvas1.create_window(20, 470, anchor="w", window=label100)

label200 = tk.Label(root, text='Model SHAP Summary Plot', font=('Arial', 14,'underline'), bg='#e9ecef')
canvas1.create_window(580, 470, anchor="w", window=label200)

image_path_1 = "SHAP-1-1.png"
image_1 = Image.open(image_path_1)
image_1 = image_1.resize((360, 240), Image.LANCZOS)
photo_1 = ImageTk.PhotoImage(image_1)

image_path_2 = "SHAP-2-2.png"
image_2 = Image.open(image_path_2)
image_2 = image_2.resize((360, 240), Image.LANCZOS)
photo_2 = ImageTk.PhotoImage(image_2)

canvas1.create_image(250,600,  image=photo_1)
canvas1.create_image(700,600,  image=photo_2)

def values():
    global New_thickness_of_aluminum_alloy_tube  # our 1st input variable
    New_thickness_of_aluminum_alloy_tube = float(entry1.get())

    global New_height_of_aluminum_alloy_tube  # our 2nd input variable
    New_height_of_aluminum_alloy_tube = float(entry2.get())

    global New_Diameter_of_aluminum_alloy_tube  # our 3rd input variable
    New_Diameter_of_aluminum_alloy_tube = float(entry3.get())

    global New_width_of_rib  # our 4th input variable
    New_width_of_rib = float(entry4.get())

    global New_length_of_rib  # our 5th input variable
    New_length_of_rib = float(entry5.get())

    global New_number_of_rib  # our 6th input variable
    New_number_of_rib = float(combo_box_1.get())

    global New_concrete_compressive_strength  # our 7th input variable
    New_concrete_compressive_strength = float(entry7.get())

    global New_temperature  # our 7th input variable
    New_temperature = float(entry8.get())


def values():
    # Validate and get the values from the entry boxes
    input_values = []
    entry_boxes = [entry1, entry2, entry3, entry4, entry5, combo_box_1, entry7, entry8]

    for entry_box in entry_boxes:
        value = entry_box.get().strip()
        if value:
            try:
                input_values.append(float(value))
            except ValueError:
                # If the value is not a valid float, show an error message or handle it as you prefer
                messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
                return
        else:
            # If any entry box is empty, show an error message or handle it as you prefer
            messagebox.showerror("Error", "Please fill in all the input fields.")
            return


    DataFrame = pd.read_excel('compress dataset.xlsx', header=[0])
    data = np.array([input_values])
    Input_data = pd.DataFrame(data, columns=['铝管直径(mm)', '铝管厚度(mm)', '铝管高度(mm)', '肋宽(mm)', '肋长(mm)', '肋数',
                                             '混凝土强度(MPa)', '温度(℃)'])

    new_data_normalized = scaler.transform(Input_data)

# Predict using the loaded Random Forest model
    Prediction_result = np.around(model_1.predict(new_data_normalized), 2)


# Display the prediction on the GUI
    label_Prediction = tk.Label(root, text=str(Prediction_result).lstrip('[').rstrip(']'), bg='#e9ecef',font=('Arial', 14, 'bold', ))
    canvas1.create_window(380, 430, anchor="w", window=label_Prediction)


# ... (rest of the GUI code)
button1 = tk.Button(root, text='Predict', command=values, bg='#e9ecef',font=('Arial', 14, 'bold', ))  # button to call the 'values' command above

canvas1.create_window(380, 390, anchor="w", window=button1)

root.mainloop()

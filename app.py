# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:04:19 2025

@author: User
"""

import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np


# ---- تحميل الموديلات ----
model1 = load_model(r"C:\Users\User\Desktop\CIFAKE_Project\Models\CIFAKE1.keras")
model2 = load_model(r"C:\Users\User\Desktop\CIFAKE_Project\Models\CIFAKE2.keras")
model3 = load_model(r"C:\Users\User\Desktop\CIFAKE_Project\Models\CIFAKE3.keras")
model4 = load_model(r"C:\Users\User\Desktop\CIFAKE_Project\Models\CIFAKE4.keras")


# عنوان رئيسي
st.title('🚀 Welcome to Beyond the Pixels')

st.subheader('🔍 Real vs AI Image Detector')


# إنشاء القائمة الجانبية
option = st.sidebar.selectbox(
    'Choose the model you want to use:',
    ('Model 1', 'Model 2', 'Model 3','Model 4')
)

# عرض النتيجة حسب الاختيار
st.write(f'✅ You Choosed : {option}')

# إضافة محتوى لكل موديل
if option == 'Model 1':
    st.write("Model One Interface ✨")
elif option == 'Model 2':
    st.write("Model Two Interface 🚀")
elif option == 'Model 3':
    st.write("Model Three Interface🔥")
elif option == 'Model 4':
    st.write("Model Four Interface 🔥")
    
    
    

# ---- رفع صورة ----
uploaded_file = st.file_uploader("📷 Upload a photo for verification", type=["jpg", "jpeg", "png"])

# ---- حسب الاختيار، حدد أي موديل تشتغل عليه ----
if option == 'Model 1':
    selected_model = model1
elif option == 'Model 2':
    selected_model = model2
elif option == 'Model 3':
    selected_model = model3
else:
    selected_model = model4

# ---- لما يرفع صورة ----
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='✅This image was uploaded', use_column_width=True)

    # تجهيز الصورة للتصنيف
    img = image.resize((32, 32))  # تأكد أن حجم الصورة يتوافق مع الموديل
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # عشان تصير 4D

    # عمل prediction
    prediction = selected_model.predict(img_array)

    # بناءً على النتيجة نحدد Fake أو Real
    if prediction[0][0] > 0.5:
        label = 'REAL ✅'
        st.success(f'The model predicts: {label} By {(prediction[0][0]*100):.2f}%')
    else:
        label = 'AI 🤖'
        st.error(f'The model predicts: {label} By {((1 - prediction[0][0])*100):.2f}%')


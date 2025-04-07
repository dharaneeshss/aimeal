import streamlit as st
import pandas as pd
import joblib
import random

# Load the trained model and preprocessors
model = joblib.load("diet_recommendation_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Define categorical and numerical features
categorical_features = ["activity_level", "region", "dietary_restriction"]
numerical_features = ["age", "height", "weight"]

# Meal recommendations database (simplified example)
meal_recommendations = {
    "Low Carb Diet": {
        "breakfast": ["Egg white omelet with vegetables", "Greek yogurt with berries", "Avocado and spinach smoothie"],
        "lunch": ["Grilled chicken salad", "Tuna lettuce wraps", "Cauliflower rice bowl with tofu"],
        "dinner": ["Baked salmon with asparagus", "Zucchini noodles with turkey meatballs", "Grilled paneer with vegetables"],
        "snacks": ["Almonds and cheese", "Celery with peanut butter", "Hard-boiled eggs"],
        "macros": {"calories": 1800, "protein": 120, "carbs": 100, "fats": 80}
    },
    "Mediterranean Diet": {
        "breakfast": ["Greek yogurt with honey and walnuts", "Whole grain toast with olive oil and tomatoes", "Vegetable frittata"],
        "lunch": ["Quinoa salad with chickpeas", "Lentil soup with whole grain bread", "Falafel wrap with tahini"],
        "dinner": ["Grilled fish with roasted vegetables", "Eggplant moussaka", "Pasta with tomato sauce and vegetables"],
        "snacks": ["Hummus with carrot sticks", "Olives and feta cheese", "Fresh fruit and nuts"],
        "macros": {"calories": 2000, "protein": 80, "carbs": 250, "fats": 70}
    },
    "Vegetarian Diet": {
        "breakfast": ["Idli with sambar", "Oatmeal with fruits and nuts", "Spinach and mushroom tofu scramble"],
        "lunch": ["Brown rice with dal", "Vegetable biryani", "Quinoa bowl with roasted vegetables"],
        "dinner": ["Grilled paneer with vegetables", "Bean and vegetable curry", "Stuffed bell peppers with rice and cheese"],
        "snacks": ["Almonds and yogurt", "Fruit smoothie", "Roasted chickpeas"],
        "macros": {"calories": 2000, "protein": 80, "carbs": 250, "fats": 70}
    }
    # Add more diet plans as needed
}

# Function to calculate BMI and recommend calories
def calculate_health_metrics(weight, height, age, activity_level):
    # Basic BMI calculation
    bmi = weight / ((height/100) ** 2)
    
    # Basic calorie estimation based on Harris-Benedict equation (simplified)
    if activity_level == "Low":
        activity_factor = 1.2
    elif activity_level == "Medium":
        activity_factor = 1.55
    else:
        activity_factor = 1.9
        
    # Base metabolic rate (simplified)
    bmr = 10 * weight + 6.25 * height - 5 * age
    calories = int(bmr * activity_factor)
    
    return bmi, calories

# Streamlit UI
st.title("AI Food Recommendation System")

# User input
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
activity_level = st.selectbox("Activity Level", ["Low", "Medium", "High"])
region = st.selectbox("Region", ["North", "South", "East", "West"])
dietary_restriction = st.selectbox("Dietary Restriction", ["None", "Vegetarian", "Vegan"])

# Create DataFrame for input
user_data = pd.DataFrame([[age, height, weight, activity_level, region, dietary_restriction]],
                         columns=numerical_features + categorical_features)

if st.button("Get Recommendations"):
    # Calculate BMI and recommended calories
    bmi, calculated_calories = calculate_health_metrics(weight, height, age, activity_level)
    
    # Encode and scale user data
    user_encoded = pd.DataFrame(ohe.transform(user_data[categorical_features]), 
                                columns=ohe.get_feature_names_out(categorical_features))
    user_scaled = pd.DataFrame(scaler.transform(user_data[numerical_features]), columns=numerical_features)
    
    # Merge processed data
    user_processed = pd.concat([user_scaled, user_encoded], axis=1)
    
    # Ensure feature names match those during training
    user_processed = user_processed.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction = model.predict(user_processed)[0]
    
    # Display result
    st.markdown("## Your Personalized Diet Plan")
    
    # Get the diet details or default to Mediterranean if not found
    diet_details = meal_recommendations.get(prediction, meal_recommendations["Mediterranean Diet"])
    
    # Randomly select meals from options
    breakfast = random.choice(diet_details["breakfast"])
    lunch = random.choice(diet_details["lunch"])
    dinner = random.choice(diet_details["dinner"])
    snacks = random.choice(diet_details["snacks"])
    
    # Adjust calories based on calculation if needed
    macros = diet_details["macros"].copy()
    if abs(calculated_calories - macros["calories"]) > 300:
        # Adjust macros proportionally
        factor = calculated_calories / macros["calories"]
        macros["calories"] = calculated_calories
        macros["protein"] = int(macros["protein"] * factor)
        macros["carbs"] = int(macros["carbs"] * factor)
        macros["fats"] = int(macros["fats"] * factor)
    
    # Display the meal plan with emojis
    st.markdown(f"""
    ✅ **Breakfast**: {breakfast}  
    ✅ **Lunch**: {lunch}  
    ✅ **Dinner**: {dinner}  
    ✅ **Snacks**: {snacks}
    """)
    
    # Display macronutrient breakdown
    st.markdown("## 🔬 Macronutrient Breakdown")
    st.markdown(f"""
    🔥 **Calories**: {macros['calories']} kcal  
    🥚 **Protein**: {macros['protein']}g  
    🍞 **Carbs**: {macros['carbs']}g  
    🥑 **Fats**: {macros['fats']}g
    """)
    
    # Show BMI information
    st.markdown("## 📊 Health Metrics")
    st.markdown(f"Your BMI: **{bmi:.1f}**")
    
    # BMI category
    bmi_category = ""
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal weight"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    st.markdown(f"BMI Category: **{bmi_category}**")
    
    # Additional recommendations based on BMI
    if bmi_category == "Underweight":
        st.info("Consider focusing on nutrient-dense foods to help reach a healthy weight.")
    elif bmi_category == "Overweight" or bmi_category == "Obese":
        st.info("This diet plan is designed to support gradual weight loss while providing essential nutrients.")
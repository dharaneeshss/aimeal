import streamlit as st
import pandas as pd
import joblib
import json
import random

# Load the trained model and preprocessors with error handling
try:
    model = joblib.load("diet_recommendation_model.pkl")
    ohe = joblib.load("ohe_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Load meal recommendations from JSON file
try:
    with open("meal_recommendations.json", "r") as f:
        meal_recommendations = json.load(f)
except Exception as e:
    st.error(f"Error loading meal recommendations: {e}")
    st.stop()

# Define categorical and numerical features
categorical_features = ["activity_level", "region", "dietary_restriction"]
numerical_features = ["age", "height", "weight"]

# Function to calculate BMI and recommend calories
def calculate_health_metrics(weight, height, age, activity_level):
    if height <= 0 or weight <= 0:
        return None, None
    bmi = weight / ((height/100) ** 2)
    if activity_level == "Low":
        activity_factor = 1.2
    elif activity_level == "Medium":
        activity_factor = 1.55
    else:
        activity_factor = 1.9
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
    if bmi is None or calculated_calories is None:
        st.error("Please enter valid height and weight values.")
        st.stop()

    # Encode and scale user data
    try:
        user_encoded = pd.DataFrame(ohe.transform(user_data[categorical_features]), 
                                    columns=ohe.get_feature_names_out(categorical_features))
        user_scaled = pd.DataFrame(scaler.transform(user_data[numerical_features]), columns=numerical_features)
    except Exception as e:
        st.error(f"Error processing input data: {e}")
        st.stop()

    # Merge processed data
    user_processed = pd.concat([user_scaled, user_encoded], axis=1)
    user_processed = user_processed.reindex(columns=feature_names, fill_value=0)

    # Make prediction
    try:
        prediction = model.predict(user_processed)[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()

    # Display result
    st.markdown("## Your Personalized Diet Plan")
    st.markdown(f"**Recommended Diet Type**: {prediction}")

    # Get the diet details or default to Balanced Diet if not found
    diet_details = meal_recommendations.get(prediction, meal_recommendations.get("Balanced Diet", {}))

    # Randomly select meals from options, handle empty lists
    def safe_choice(meal_list):
        return random.choice(meal_list) if meal_list else "No meal available"

    breakfast = safe_choice(diet_details.get("breakfast", []))
    lunch = safe_choice(diet_details.get("lunch", []))
    dinner = safe_choice(diet_details.get("dinner", []))
    snacks = safe_choice(diet_details.get("snacks", []))

    # Adjust calories based on calculation if needed
    macros = diet_details.get("macros", {"calories": 0, "protein": 0, "carbs": 0, "fats": 0}).copy()
    if macros["calories"] and abs(calculated_calories - macros["calories"]) > 300:
        factor = calculated_calories / macros["calories"] if macros["calories"] else 1
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
        st.info("This diet plan is designed to support gradual weight loss while providing essential nutrients. Consider consulting a healthcare provider for personalized advice.")

    # Save user input data for future analysis
    try:
        with open("user_data_log.csv", "a") as f:
            user_data.to_csv(f, header=f.tell()==0, index=False)
    except Exception as e:
        st.warning(f"Error saving user data: {e}")

    st.success("Recommendation generated successfully!")
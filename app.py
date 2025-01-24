import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px


st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("🌾 Crop Yield Prediction & Recommendation System")
st.markdown("""
This application uses machine learning to predict crop yields based on various environmental 
and agricultural factors. The model has been trained on historical crop data with an R² score 
of 0.979 (97.9% accuracy).
""")

def load_data():
    """Load the model and necessary encoders"""
    models = {}
    try:
        for model_name in ["random_forest_model", "linear_regression_model", "decision_tree_model", 
                           "gradient_boosting_model", "support_vector_model", "k-nearest_neighbors_model"]:
            with open(f'{model_name}.pkl', 'rb') as f:
                models[model_name] = pickle.load(f)
        
        crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))
        season_encoder = pickle.load(open('season_encoder.pkl', 'rb'))
        state_encoder = pickle.load(open('state_encoder.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        feature_info = pickle.load(open('feature_info.pkl', 'rb'))
        
        return models, crop_encoder, season_encoder, state_encoder, scaler, feature_info
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None, None, None, None, None

def make_prediction(model, crop, season, state, area, rainfall, fertilizer, pesticide, crop_encoder, season_encoder, state_encoder, scaler):
    """Make prediction using the loaded model"""
    try:
       
        crop_encoded = crop_encoder.transform([crop])[0]
        season_encoded = season_encoder.transform([season])[0]
        state_encoded = state_encoder.transform([state])[0]
        
      
        numerical_features = np.array([[area, rainfall, fertilizer, pesticide]])
        numerical_scaled = scaler.transform(numerical_features)
        
       
        features = np.column_stack([
            crop_encoded, 
            season_encoded, 
            state_encoded, 
            numerical_scaled
        ])
        
       
        prediction = model.predict(features)[0]
        
        return prediction
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def create_performance_plot(performance_metrics):
    """Create a performance comparison plot for all models"""
    model_names = list(performance_metrics.keys())
    r2_scores = [metrics['R2 Score'] for metrics in performance_metrics.values()]
    rmse_values = [metrics['RMSE'] for metrics in performance_metrics.values()]
    mae_values = [metrics['MAE'] for metrics in performance_metrics.values()]
    mse_values = [metrics['MSE'] for metrics in performance_metrics.values()]

   
    fig_r2 = px.bar(
        x=model_names,
        y=r2_scores,
        title='Model Performance Comparison (R² Score)',
        labels={'x': 'Models', 'y': 'R² Score'},
        color=r2_scores,
        color_continuous_scale='Viridis'
    )
    
    
    fig_rmse = px.bar(
        x=model_names,
        y=rmse_values,
        title='Model Performance Comparison (RMSE)',
        labels={'x': 'Models', 'y': 'RMSE'},
        color=rmse_values,
        color_continuous_scale='Blues'
    )
    
  
    fig_mae = px.bar(
        x=model_names,
        y=mae_values,
        title='Model Performance Comparison (MAE)',
        labels={'x': 'Models', 'y': 'MAE'},
        color=mae_values,
        color_continuous_scale='Reds'
    )
    
    
    fig_mse = px.bar(
        x=model_names,
        y=mse_values,
        title='Model Performance Comparison (MSE)',
        labels={'x': 'Models', 'y': 'MSE'},
        color=mse_values,
        color_continuous_scale='Greens'
    )
    
    return fig_r2, fig_rmse, fig_mae, fig_mse

def recommend_top_crops(models, crop_encoder, season_encoder, state_encoder, scaler, feature_info, selected_model, season, state, area, rainfall, fertilizer, pesticide, top_n=5):
    """Recommend the top N crops based on the highest predicted yield and display results in a table."""
   
    crop_yield_predictions = []

    
    for crop in feature_info['crops']:
       
        crop_encoded = crop_encoder.transform([crop])[0]
        season_encoded = season_encoder.transform([season])[0]
        state_encoded = state_encoder.transform([state])[0]
        
       
        numerical_features = np.array([[area, rainfall, fertilizer, pesticide]])
        numerical_scaled = scaler.transform(numerical_features)
        
        
        features = np.column_stack([[crop_encoded], [season_encoded], [state_encoded], numerical_scaled])
        
       
        predicted_yield = models[selected_model].predict(features)[0]
        
        crop_yield_predictions.append((crop, predicted_yield))
    
    
    top_crops = sorted(crop_yield_predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
  
    if top_crops:
        st.info("Top Crop Recommendations Based on Predicted Yield:")
        
       
        df_recommendations = pd.DataFrame(top_crops, columns=["Crop", "Predicted Yield (tons/hectare)"])
        st.table(df_recommendations)
    else:
        st.warning("No crop recommendation could be generated.")






def main():
    
    models, crop_encoder, season_encoder, state_encoder, scaler, feature_info = load_data()
    
    
    if not all([models, crop_encoder, season_encoder, state_encoder, scaler, feature_info]):
        st.error("One or more required files are missing. Please verify all model files are present.")
        return
   
   
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ("Prediction", "Comparison","Recommendation"))

    
    if page == "Prediction":
        st.subheader("Prediction")

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Input Parameters")
            
           
            crop = st.selectbox('Select Crop', feature_info['crops'])
            season = st.selectbox('Select Season', feature_info['seasons'])
            state = st.selectbox('Select State', feature_info['states'])
            
            area = st.number_input('Area (hectares)', 
                min_value=0.0, max_value=1000000.0, value=100.0)
            
            rainfall = st.number_input('Annual Rainfall (mm)', 
                min_value=0.0, max_value=5000.0, value=1000.0)
            
            fertilizer = st.number_input('Fertilizer (kg)', 
                min_value=0.0, max_value=1000000.0, value=1000.0)
            
            pesticide = st.number_input('Pesticide (kg)', 
                min_value=0.0, max_value=1000000.0, value=100.0)
            
            model_name = st.selectbox('Select Model', list(models.keys()))
            predict_button = st.button('Predict Yield')
        
        with col2:
            if predict_button:
                
                prediction = make_prediction(models[model_name], crop, season, state, area, rainfall, fertilizer, pesticide, crop_encoder, season_encoder, state_encoder, scaler)
                
                if prediction is not None:
                    
                    st.subheader("Prediction Results")
                    st.markdown(f"### Predicted Yield: {prediction:.2f} tons/hectare")
                    
                  
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Yield per Square Meter", value=f"{(prediction/10000):.3f} tons")
                    with col2:
                        st.metric(label="Yield per Acre", value=f"{(prediction*0.404686):.2f} tons")
                    with col3:
                        st.metric(label="Total Expected Yield", value=f"{(prediction*area):.2f} tons")
                    
                    
                    st.subheader("Input Summary")
                    input_data = pd.DataFrame({
                        'Parameter': ['Crop', 'Season', 'State', 'Area', 'Rainfall', 'Fertilizer', 'Pesticide'],
                        'Value': [crop, season, state, f"{area:.2f} ha", f"{rainfall:.2f} mm", f"{fertilizer:.2f} kg", f"{pesticide:.2f} kg"]
                    })
                    st.table(input_data)
                    
                    if model_name in feature_info['performance_metrics']:
                        model_r2 = feature_info['performance_metrics'][model_name]['R2 Score']
                        model_rmse = feature_info['performance_metrics'][model_name]['RMSE']
                        MODEL_MAE=feature_info['performance_metrics'][model_name]['MAE']
                        MODEL_MSE=feature_info['performance_metrics'][model_name]['MSE']
                        st.info(f"**Model Confidence Metrics:**\n- R² Score: {model_r2:.3f}\n- RMSE: {model_rmse:.2f} tons/hectare \n- MAE: {MODEL_MAE:.2f} tons/hectare \n- MSE: {MODEL_MSE:.2f} tons/hectare")
                    else:
                        st.warning(f"Performance metrics not available for the selected model: {model_name}")

  
    elif page == "Comparison":
        st.subheader("Model Performance Comparison")
        r2_plot, rmse_plot, mae_plot, mse_plot = create_performance_plot(feature_info['performance_metrics'])
        st.plotly_chart(r2_plot)
        st.plotly_chart(rmse_plot)
        st.plotly_chart(mae_plot)
        st.plotly_chart(mse_plot)
        
        
        performance_data = pd.DataFrame.from_dict(feature_info['performance_metrics'], orient='index')
        st.subheader("Performance Metrics Summary")
        st.table(performance_data)

    elif page == "Recommendation":
       
        selected_model = max(feature_info['performance_metrics'], key=lambda model: feature_info['performance_metrics'][model]['R2 Score'])

        st.info(f"**Model Selection:** Based on maximum R² score, the recommended model is **{selected_model}**.")
        
        season = st.selectbox("Season", feature_info['seasons'])
        state = st.selectbox("State", feature_info['states'])
        area = st.number_input("Area (in hectares)", min_value=0.1)
        rainfall = st.number_input("Rainfall (in mm)", min_value=0.0)
        fertilizer = st.number_input("Fertilizer Usage (in kg)", min_value=0.0)
        pesticide = st.number_input("Pesticide Usage (in kg)", min_value=0.0)
        
        if st.button("Get Top 5 Crop Recommendations"):
            recommend_top_crops(
                models, crop_encoder, season_encoder, state_encoder, scaler, 
                feature_info, selected_model, season, state, area, rainfall, fertilizer, pesticide, top_n=5
            )

    
    st.sidebar.header("About")
    st.sidebar.info("""This app predicts crop yields using various regression models trained on historical crop data.""")

if __name__ == "__main__":
    main()




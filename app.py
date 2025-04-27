import streamlit as st
import io
import base64
import matplotlib.pyplot as plt
from userMachine import run_model  # make sure userMachine.py is present

# Streamlit page setup
st.set_page_config(page_title="WhoTheGOAT - MVP Predictor", layout="centered")

st.title("🏀 WhoTheGOAT - NBA MVP Predictor")

st.write("Assign a percentage weight (must add to 100%) to the following stats:")

# Create a form for user input
with st.form("input_form"):
    pts = st.number_input("Points (PTS) %", min_value=0.0, max_value=100.0, value=0.0)
    ast = st.number_input("Assists (AST) %", min_value=0.0, max_value=100.0, value=0.0)
    trb = st.number_input("Rebounds (TRB) %", min_value=0.0, max_value=100.0, value=0.0)
    blk = st.number_input("Blocks (BLK) %", min_value=0.0, max_value=100.0, value=0.0)

    submit_button = st.form_submit_button(label="Predict MVPs")

# If the button is clicked
if submit_button:
    user_input_percentages = {
        'PTS': pts,
        'AST': ast,
        'TRB': trb,
        'BLK': blk
    }

    # Validate total
    if sum(user_input_percentages.values()) != 100:
        st.error("❌ The total of percentages must equal 100. Please adjust.")
    else:
        try:
            # Run the model
            fig_dict = run_model(user_input_percentages)

            st.success("✅ Prediction successful! Here are the results:")

            # Display the top 5 MVP figure
            st.subheader("Top 5 Predicted MVPs 📈")
            st.pyplot(fig_dict['top_5_mvp'])

            # (Optional) Display other plots if you want
            with st.expander("See Feature Importance Chart"):
                st.pyplot(fig_dict['feature_importance'])

            with st.expander("See Error Metrics Chart"):
                st.pyplot(fig_dict['error_metrics'])

            with st.expander("See R² Backtest Over Time"):
                st.pyplot(fig_dict['r2_backtest'])

        except Exception as e:
            st.error(f"An error occurred: {e}")

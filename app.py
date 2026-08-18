import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Car Valuation Tool", page_icon="🚗", layout="centered"
)
st.title("🚗 Car Valuation")
st.markdown("Estimate market price via FastAPI backend service.")


try:
    options_res = requests.get(f"{API_URL}/options", timeout=3)
    options = options_res.json()
except Exception:
    st.error(
        "Cannot connect to the FastAPI backend. Ensure `uvicorn main:app --reload` is running on port 8000."
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Car Brand / Make", options["makes"])
    model_name = st.text_input(
        "Model Name (e.g., Civic, Corolla, Alto, Mehran)", value="Corolla"
    )
    model_year = st.slider(
        "Model Year", min_value=1990, max_value=2026, value=2019
    )
    mileage = st.number_input(
        "Mileage (KM)",
        min_value=100,
        max_value=500000,
        value=65000,
        step=5000,
    )

with col2:
    engine_capacity = st.number_input(
        "Engine Displacement (cc)",
        min_value=600,
        max_value=6000,
        value=1600,
        step=100,
    )
    transmission = st.selectbox("Transmission", options["transmissions"])
    fuel_type = st.selectbox("Fuel Type", options["fuel_types"])

    has_auction = st.checkbox("Has Japanese Auction Rating?")
    auction_rating = (
        st.slider("Auction Rating", 1.0, 10.0, 4.0, 0.5)
        if has_auction
        else -1.0
    )

if st.button("Predict Valuation", type="primary", use_container_width=True):
    payload = {
        "make": make,
        "model_name": model_name,
        "model_year": model_year,
        "mileage": mileage,
        "engine_capacity": engine_capacity,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "auction_rating": auction_rating,
    }

    with st.spinner("Calculating estimate..."):
        try:
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                price = data["estimated_price_pkr"]
                lacs = data["price_in_lacs"]
                crores = data["price_in_crores"]

                st.markdown("---")
                st.success(f"### Estimated Price: **PKR {price:,.0f}**")

                if crores >= 1.0:
                    st.info(f"Market Scale: **~{crores:.2f} Crore PKR**")
                else:
                    st.info(f"Market Scale: **~{lacs:.2f} Lac PKR**")
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
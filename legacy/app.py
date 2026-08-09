import streamlit as st
import datetime
import requests
from PIL import Image
import pandas as pd
from mainpy import plan_my_trip
from mainpy import direct_flights, via_nearby_airport, direct_trains,multi_leg_train, train_partial_road, bus_or_cab_only,get_weather_forecast_node
from mainpy import human_node, wikivoyage_daywise_plan,search_hotels_serpapi,get_etrain_trains
from mainpy import llm_decider,find_direct_flights,find_cabs,extract_hub_city_names,search_flights_tool
from mainpy import dynamic_breakup_journey,dynamic_breakup_journey_cab,extract_arrival_time,should_continue

from airplane_data import get_airport_name,parse_iso_time,calculate_layovers
from mainpy import get_abhibus_buses,find_cabs
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable

from langgraph.graph import StateGraph, END

from typing import Annotated, List, Optional
from langgraph.graph import StateGraph
from langchain.schema import BaseMessage
from typing import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    parsed: dict
    decision: Annotated[Optional[str], "Which travel strategy to use"]
    arrival_time: Annotated[Optional[str], "Estimated arrival time"]
    transportation_summary:Annotated[Optional[str],"Final Transport plan"]
    weather_data:list
    hotels_info:Optional[str]
    # estimated_budget:Optional[int]
    user_feedback:Optional[str]
    human_review_result:Optional[str]
    llm_routing_decision:Optional[str]
    human_feedback:Optional[str]
    plan:Optional[str]


from langgraph.graph import StateGraph
graph_builder = StateGraph(State)

from mainpy import build_trip_graph

from langchain_groq import ChatGroq
llm=ChatGroq(model="llama3-70b-8192")

# Set page config
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    
    /* Increase sidebar width */
    [data-testid="stSidebar"] {
        min-width: 500px;
        max-width: 5000px;
    }

    /* Increase sidebar font size */
    [data-testid="stSidebar"] * {
        font-size: clamp(16px, 1.2vw, 22px);
    }

    /* Optional: add some padding for a spacious feel */
    [data-testid="stSidebar"] {
        padding: 1.5rem;
    }

    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        border-radius: 5px;
    }
    .stDateInput>div>div>input {
        border-radius: 5px;
    }
    .stNumberInput>div>div>input {
        border-radius: 5px;
    }
    .travel-plan {
        background: rgba(255, 255, 255, 0.05);;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .weather-card {
        background: rgba(255, 255, 255, 0.05);;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .hotel-card {
        background: rgba(255, 255, 255, 0.05);;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .transport-card {
        background: rgba(255, 255, 255, 0.05);;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
   .title {
    font-size: 50px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 20px;
}
    .mark{
    font-size:25px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<div class = "title">✈️ AI Travel Planner</div>',unsafe_allow_html=True)
st.markdown('<div class ="mark">Plan your perfect trip with our AI-powered travel assistant. Get personalized recommendations for transportation,accommodation, and activities based on your preferences.</div>',unsafe_allow_html=True)

# Sidebar with user inputs
with st.sidebar:
    st.header("Trip Details")
    
    # Origin and Destination
    origin = st.text_input("Origin City", "Delhi")
    destination = st.text_input("Destination City", "Mumbai")
    
    # Travel dates
    today = datetime.date.today()
    start_date = st.date_input("Start Date", today)
    end_date = st.date_input("End Date", today + datetime.timedelta(days=7))
    
    # Number of travelers
    num_travelers = st.number_input("Number of Travelers", min_value=1, max_value=10, value=2)
    
    # Travel preferences
    travel_class = st.selectbox("Travel Class", ["Economy", "Business", "First Class"])
    budget = st.selectbox("Budget Range", ["Low", "Medium", "High"])
    # Collect selected interests into a list
    interests = []
    # Interests checkboxes
    st.markdown("**Interests (Select all that apply)**")
    col1, col2 = st.columns(2)
    with col1:
        sightseeing = st.checkbox("Sightseeing", False)
        adventure = st.checkbox("Adventure")
        beaches = st.checkbox("Beaches")
    with col2:
        shopping = st.checkbox("Shopping")
        food = st.checkbox("Food & Dining", False)
        culture = st.checkbox("Culture & History")
    
    if sightseeing:
        interests.append("Sightseeing")
    if adventure:
        interests.append("Adventure")
    if beaches:
        interests.append("Beaches")
    if shopping:
        interests.append("Shopping")
    if food:
        interests.append("Food & Dining")
    if culture:
        interests.append("Culture & History")
    
    # Additional preferences
    special_requirements = st.text_area("Special Requirements", "e.g., wheelchair accessible, pet-friendly")
    
    # Submit button
    if st.button("Plan My Trip"):
        st.session_state['trip_submitted'] = True

# Main content area
if 'trip_submitted' in st.session_state and st.session_state['trip_submitted']:
    st.success("Trip details received! Generating your personalized travel plan...")

    # Simulate loading while the AI processes the request
    with st.spinner("Analyzing weather data, finding best routes, and selecting accommodations..."):
        # Simulate API call delay
        graph=build_trip_graph()
        plan_my_trip(origin, destination, start_date,end_date, num_travelers, interests,graph)

        transport_summary = st.session_state["state"].get("transportation_summary", "Transport plan not available")
        plan_text = st.session_state["state"].get("plan", "Plan not generated")
        hotel_info = st.session_state["state"].get("hotels_info", "No hotel info found")
        weather_info=st.session_state["state"].get("weather_data")

        import time
        time.sleep(3)
        
        # Display trip summary
        st.subheader("Your Trip Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Origin", origin)
        with col2:
            st.metric("Destination", destination)
        with col3:
            duration = (end_date - start_date).days
            st.metric("Duration", f"{duration} days")
        
        # Weather forecast section
        st.subheader("🌤️ Weather Forecast")
        with st.expander("🌤️ Weather Forecast"):
            for day in weather_info:
                weather_card = f"""
                <div style="
                    background: rgba(255, 255, 255, 0.05);;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                ">
                    <h4 style="margin-bottom: 5px;">📅 {day['day_name']}, {day['date']}</h4>
                    <p>🌦️ <strong>Condition:</strong> {day['condition']}</p>
                    <p>🌡️ <strong>Max:</strong> {day['max_temp']}°C &nbsp;&nbsp; 
                    <strong>Min:</strong> {day['min_temp']}°C &nbsp;&nbsp; 
                    <strong>Avg:</strong> {day['avg_temp']}°C</p>
                    <p>💧 <strong>Humidity:</strong> {day['humidity']}% &nbsp;&nbsp;
                    🌧️ <strong>Rain Chance:</strong> {day['rain_chance']}%</p>
                </div>
                """
                st.markdown(weather_card, unsafe_allow_html=True)
        
        # Transportation options
        st.subheader("🚗 Transportation Options")
        with st.expander("🚗 Transportation Options"):
            st.markdown(f"""
            <div class="transport-card">
            {st.session_state["state"].get("transportation_summary")}
            </div>
            """, unsafe_allow_html=True)
        
        # Hotel recommendations
        st.subheader("🏨 Hotel Recommendations")
        with st.expander("🏨 Hotel Recommendations"):
            st.markdown(f"""
            <div class="hotel-card">
                {st.session_state["state"].get("hotels_info", "No hotel info available.")}
            </div>
            """, unsafe_allow_html=True)
        
        # Daily itinerary
        st.subheader("📅 Daily Itinerary")
        
        # Day 1
        with st.expander("Daywise Plan"):
            st.markdown(f"""
            <div class="travel-plan">
                {st.session_state["state"].get("plan")}
            </div>
            """, unsafe_allow_html=True)

        # Budget summary
        st.subheader("💰 Estimated Budget")
        budget_data = {
            "Category": ["Flights", "Hotels (3 nights)", "Local Transport", "Meals", "Activities", "Miscellaneous"],
            "Estimated Cost (₹)": ["10,000", "36,000", "5,000", "12,000", "8,000", "5,000"],
            "Per Person (₹)": ["5,000", "18,000", "2,500", "6,000", "4,000", "2,500"]
        }
        budget_df = pd.DataFrame(budget_data)
        st.dataframe(budget_df, hide_index=True, use_container_width=True)
        
        st.markdown("**Total Estimated Cost: ₹76,000** (₹38,000 per person)")
        
        # Download button for the itinerary
        st.download_button(
            label="Download Itinerary (PDF)",
            data="This would be the PDF content in a real app",
            file_name=f"{origin}_to_{destination}_itinerary.pdf",
            mime="application/pdf"
        )
        
        # Feedback section
        st.subheader("📝 Feedback")
        feedback = st.text_area("How can we improve your travel plan?")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback! We'll use it to improve our service.")
else:
    # Default view before trip is planned
    st.info("Please fill in your trip details in the sidebar and click 'Plan My Trip' to get started.")
    
    # Sample travel destinations
    st.subheader("Popular Destinations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1646748019039-e908f7e41282?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8Z29hJTIwYmVhY2h8ZW58MHx8MHx8fDA%3D", 
                 caption="Goa - Beaches & Nightlife", use_container_width=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1587474260584-136574528ed5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80", 
                 caption="Rajasthan - Culture & Heritage", use_container_width=True)
    
    with col3:
        st.image("https://images.unsplash.com/photo-1593693397690-362cb9666fc2?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8a2VyYWxhfGVufDB8fDB8fHww", 
                 caption="Kerala - Backwaters & Nature", use_container_width=True)
    
    # Testimonials
    st.subheader("What Our Users Say")
    testimonial1, testimonial2 = st.columns(2)
    
    with testimonial1:
       st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;">
        <p style="font-size: 20px;">"The AI Travel Planner saved me hours of research! The itinerary was perfectly tailored to my interests."</p>
        <p style="font-weight: bold; margin-top: 10px;">- Priya S., Bangalore</p>
    </div>
""", unsafe_allow_html=True)


    
    with testimonial2:
        st.markdown("""
        <div style="
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;">
        <p style="font-size : 20px;">"I never would have found those hidden gems without this tool. The transportation recommendations were spot on!"</p>
        <p style = "font-weight : bold;margin-top: 10px;">- Rajiv M., Delhi</p>
        </div>
        """, unsafe_allow_html=True)

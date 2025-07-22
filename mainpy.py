# %%
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import TextSplitter
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated,Optional
from typing_extensions import TypedDict
from typing import Annotated, List, Optional
from langgraph.graph import StateGraph
from langchain.schema import BaseMessage

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


# from langchain_groq import ChatGroq
# llm=ChatGroq(model="llama3-70b-8192")

# %%
from trip_extractor import extract_trip_fields
from bus_data import get_abhibus_buses
from cab_data import find_cabs
from train_data import get_etrain_trains
from airplane_data import search_flights_tool,get_airport_name,parse_iso_time,calculate_layovers
from hotels_data import search_hotels_serpapi2
from daywise_plan import generate_daywise_plan

# %%
import requests
from langchain.tools import Tool
from datetime import datetime, timedelta
from langchain.schema import AIMessage

WEATHER_API_KEY = "6bba134b08cb4762946190226252306"

def get_weather_forecast_node(state: State) -> State:
    """
    A LangGraph-compatible node that reads city, start_date, and duration from the parsed state,
    calls the weather API, and appends the forecast as an AIMessage.
    Also stores structured weather data for use by daywise_plan node.
    """
    parsed = state.get("parsed", {})
    messages = state.get("messages", [])
    
    try:
        city = parsed.get("destination")  # Or use 'origin' if you prefer
        start_date_str = parsed.get("start_date")
        num_days = int(parsed.get("duration", 3))  # Default to 3 days if not available

        if not city or not start_date_str:
            raise ValueError("City or start date missing in parsed state")

        # Validate date
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        today = datetime.today().date()

        results = []
        weather_data = []  # Store structured weather data for daywise_plan
        
        for i in range(min(num_days, 14)):
            target_date = start_date + timedelta(days=i)
            endpoint = "forecast.json" if target_date >= today else "history.json"
            url = f"http://api.weatherapi.com/v1/{endpoint}"
            params = {
                "key": WEATHER_API_KEY,
                "q": city,
                "dt": target_date.strftime("%Y-%m-%d")
            }

            try:
                res = requests.get(url, params=params, timeout=5)
                res.raise_for_status()
                data = res.json()
            except requests.exceptions.RequestException as re:
                return {
                    "messages": messages + [AIMessage(content=f"❌ Weather API error: {str(re)}")],
                    "parsed": parsed
                }

            if "error" in data:
                return {
                    "messages": messages + [AIMessage(content=f"❌ API Error on {target_date}: {data['error']['message']}")],
                    "parsed": parsed
                }

            forecast_days = data.get("forecast", {}).get("forecastday", [])
            if not forecast_days:
                return {
                    "messages": messages + [AIMessage(content=f"❌ No forecast data for {target_date}")],
                    "parsed": parsed
                }

            day_info = forecast_days[0]['day']
            condition = day_info['condition']['text']
            max_temp = day_info['maxtemp_c']
            min_temp = day_info['mintemp_c']

            # For display message
            results.append(f"📅 {target_date.strftime('%b %d, %Y')}: {condition}, 🌡️ {min_temp}°C to {max_temp}°C")

            # Store structured data for daywise_plan node
            weather_data.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "day_name": target_date.strftime("%A"),
                "condition": condition,
                "max_temp": max_temp,
                "min_temp": min_temp,
                "avg_temp": day_info.get('avgtemp_c', (max_temp + min_temp) / 2),
                "rain_chance": day_info.get('daily_chance_of_rain', 0),
                "snow_chance": day_info.get('daily_chance_of_snow', 0),
                "humidity": day_info.get('avghumidity', 0)
            })

        forecast_message = f"📍 **Weather Forecast for {city}**\n\n" + "\n".join(results)
        st.subheader("🌤️ Weather Forecast")
        with st.expander("🌤️ Weather Forecast"):
            for day in weather_data:
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
        
        return {
            "messages": messages + [AIMessage(content=forecast_message)],
            "weather_data": weather_data  # Structured data for daywise_plan
        }
    

    except Exception as e:
        return {
            "messages": messages + [AIMessage(content=f"❌ Weather Forecast Node Error: {str(e)}")],
            "weather_data": []  # Empty weather data on error
        }

# %%
def search_hotels_serpapi(state: State) -> State:
    try:
        parsed = state['parsed']
        
        feedback=state.get("user_feedback","")
        st.write("🔍 Hotel Search Input", {
            "destination": parsed.get("destination"),
            "start_date": parsed.get("start_date"),
            "end_date": parsed.get("end_date"),
            "duration": parsed.get("duration"),
            "persons": int(parsed.get("Persons"))
        })

        hotels = search_hotels_serpapi2(parsed["destination"], parsed["start_date"], parsed["end_date"],parsed["duration"],feedback,int(parsed["Persons"]))
        hotel_info = f"🏨 **Hotels in {parsed['destination']}**:\n{hotels}"

        st.subheader("🏨 Hotel Recommendations")
        with st.expander("🏨 Hotel Recommendations"):
            st.markdown(f"""
            <div class="hotel-card">
                {hotel_info}
            </div>
            """, unsafe_allow_html=True)

        return {
            "messages": [*state["messages"], AIMessage(content=hotel_info)],
            "parsed": parsed,
            "decision": state.get("decision"),
            "hotels_info":hotel_info
        }
    except Exception as e:
        error_msg = f"❌ Hotels Tool failed: {e}"
        st.warning(error_msg)
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision")
        }

# %%
from langchain.schema import AIMessage

def wikivoyage_daywise_plan(state: State) -> State:
    """Uses WikiVoyage to generate a day-wise itinerary"""
    user_input = state["messages"][-1].content
    tour_info = []
    
    try:
        # STEP 1: Extract trip info
        parsed = state['parsed']
        destination = parsed["destination"]
        country = "India"
        interest = parsed.get("preferences", "sightseeing")
        weather_data = state.get("weather_data", [])
        duration = parsed.get("duration", "3")  # Default to 3 days if not specified
        arrival_time=state.get("arrival_time","14:00")
        arrival_date=parsed.get("start_date")
        hotel_info=state.get("hotels_info")
        persons = int(parsed.get("Persons", 1))
        feedback=state.get("human_feedback","")
        
        # estimated_budget_travel=int(state.get("estimated_budget",0))
        # print(estimated_budget_travel)

    except Exception as e:
        return {
            "messages": [*state["messages"], AIMessage(content=f"❌ Failed to extract trip fields: {e}")]
        }
    
    try:
        # STEP 2: Get the WikiVoyage plan using a real function (not recursive call!)
        plan = generate_daywise_plan(user_input,destination,interest,weather_data,duration,arrival_time,arrival_date,hotel_info,persons,feedback)  # <-- Use actual planning function here
        
        # print(budget)

        # total_budget = estimated_budget_travel + budget

        tour_info.append(f"🧭 **Itinerary**:\n{plan}")
    except Exception as e:
        tour_info.append(f"❌ WikiVoyage Tool failed: {e}")
    
    combined_output = "\n\n".join(tour_info)
    
    st.subheader("📅 Daily Itinerary")
        
        # Day 1
    with st.expander("Daywise Plan"):
        st.markdown(f"""
        <div class="travel-plan">
            {combined_output}
        </div>
        """, unsafe_allow_html=True)

    return {
        "messages": [*state["messages"], AIMessage(content=combined_output)],
        "plan":combined_output
        # "estimated_budget":total_budget
    }

# %%
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import re
from typing import List
from typing_extensions import NotRequired, TypedDict
from langchain.schema import BaseMessage


# Remove @tool decorator and make these regular functions that work with state
def direct_flights(state: State) -> State:
    """Get direct flights between origin and destination"""
    try:
        parsed = state['parsed']
        tool_outputs = []
        flights = search_flights_tool(
            origin=parsed["origin"],
            destination=parsed["destination"],
            departure_date=parsed["start_date"],
            return_date=parsed["end_date"],
            adults = int(parsed.get("Persons", 1)), # Default to 1 if not provided
            currency="INR",
            src_IATA=parsed["src_IATA"],
            dest_IATA=parsed["dest_IATA"],
            max_results=5
        )
        if not flights:
            raise ValueError("No flights returned by search_flights_tool.")

        tool_outputs.append(f"✈️ **Direct Flights from {parsed['origin']} to {parsed['destination']}**:\n{flights}")
        print("Tools Output : ",tool_outputs)
        combined = "\n\n".join([str(x) for x in tool_outputs])
        return {
            "messages": [*state["messages"], AIMessage(content=combined)],

            "decision": state.get("decision") 
        }

    except Exception as e:
        error_msg = f"❌ Direct Flights Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],

            "decision": state.get("decision") 
        }

def via_nearby_airport(state: State) -> State:
    """Get travel options via nearby airports with connecting transportation"""
    try:
        parsed = state['parsed']
        tool_outputs = []
        
        # Check if we need ground transport to nearby source airport
        if parsed['origin'] != parsed['nearby_src_city']:
            tool_outputs.append(f"🚗 **Ground transport from {parsed['origin']} to {parsed['nearby_src_city']}**:")
            
            # Get ground transport options
            date_ddmmyyyy = datetime.strptime(parsed["start_date"], "%Y-%m-%d").strftime("%d/%m/%Y")
            
            # Buses
            try:
                buses = get_abhibus_buses(parsed["origin"], parsed["nearby_src_city"], date_ddmmyyyy)
                tool_outputs.append(f"🚌 **Buses**: {buses}")
            except Exception as e:
                tool_outputs.append(f"❌ Buses failed: {e}")
            
            # Trains
            try:
                trains = get_etrain_trains(parsed["origin"], parsed["nearby_src_city"], date_ddmmyyyy)
                if isinstance(trains, str) and "Empty DataFrame" in trains:
                    tool_outputs.append("🚆 **Trains**: ❌ No direct trains found")
                else:
                    tool_outputs.append(f"🚆 **Trains**: {trains}")
            except Exception as e:
                tool_outputs.append(f"❌ Trains failed: {e}")
            
            # Cabs
            try:
                cabs = find_cabs(parsed["origin"], parsed["nearby_src_city"], parsed["start_date"],parsed["Persons"])
                tool_outputs.append(f"🚖 **Cabs**: {cabs}")
            except Exception as e:
                tool_outputs.append(f"❌ Cabs failed: {e}")
        
        # Main flight between nearby airports
        try:
            flights = search_flights_tool(
                origin=parsed["nearby_src_city"],
                destination=parsed["nearby_dest_city"],
                departure_date=parsed["start_date"],
                return_date=parsed["end_date"],
                adults = int(parsed.get("Persons", 1)), # Default to 1 if not provided
                currency="INR",
                src_IATA=parsed["src_IATA"],
                dest_IATA=parsed["dest_IATA"],
                max_results=5
            )
            tool_outputs.append(f"✈️ **Main Flight ({parsed['nearby_src_city']} → {parsed['nearby_dest_city']})**:\n{flights}")
        except Exception as e:
            tool_outputs.append(f"❌ Main flight failed: {e}")
        
        # Check if we need ground transport from nearby destination airport
        if parsed['destination'] != parsed['nearby_dest_city']:
            tool_outputs.append(f"🚗 **Ground transport from {parsed['nearby_dest_city']} to {parsed['destination']}**:")
            
            date_ddmmyyyy = datetime.strptime(parsed["start_date"], "%Y-%m-%d").strftime("%d/%m/%Y")
            
            # Buses
            try:
                buses = get_abhibus_buses(parsed["nearby_dest_city"], parsed["destination"], date_ddmmyyyy)
                tool_outputs.append(f"🚌 **Buses**: {buses}")
            except Exception as e:
                tool_outputs.append(f"❌ Buses failed: {e}")
            
            # Trains
            try:
                trains = get_etrain_trains(parsed["nearby_dest_city"], parsed["destination"], date_ddmmyyyy)
                if isinstance(trains, str) and "Empty DataFrame" in trains:
                    tool_outputs.append("🚆 **Trains**: ❌ No direct trains found")
                else:
                    tool_outputs.append(f"🚆 **Trains**: {trains}")
            except Exception as e:
                tool_outputs.append(f"❌ Trains failed: {e}")
            
            # Cabs
            try:
                cabs = find_cabs(parsed["nearby_dest_city"], parsed["destination"], parsed["start_date"],parsed["Persons"])
                tool_outputs.append(f"🚖 **Cabs**: {cabs}")
            except Exception as e:
                tool_outputs.append(f"❌ Cabs failed: {e}")
        
        combined = "\n\n".join([str(x) for x in tool_outputs])
        return {
            "messages": [*state["messages"], AIMessage(content=combined)],
            "decision": state.get("decision") 
        }
            
    except Exception as e:
        error_msg = f"❌ Via Nearby Airport Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision") 
        }

def direct_trains(state: State) -> State:
    """Get direct train options between origin and destination"""
    try:
        parsed = state['parsed']
        date_ddmmyyyy = datetime.strptime(parsed["start_date"], "%Y-%m-%d").strftime("%d/%m/%Y")

        trains = get_etrain_trains(parsed["origin"], parsed["destination"], date_ddmmyyyy)
        
        if isinstance(trains, list):
            top_trains = trains[:8]
        else:
            top_trains = trains.head(8) if hasattr(trains, 'head') else trains
        
        output = f"🚆 **Direct Trains from {parsed['origin']} to {parsed['destination']}**:\n{top_trains}"
        
        return {
            "messages": [*state["messages"], AIMessage(content=output)],
            "decision": state.get("decision") 
        }
    
    except Exception as e:
        error_msg = f"❌ Direct Trains Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision") 
        }

def find_direct_flights(parsed: dict) -> dict:
    """Helper to check if direct flight is possible"""
    has_direct = (
        parsed.get('origin') == parsed.get('nearby_src_city') and 
        parsed.get('destination') == parsed.get('nearby_dest_city') and
        parsed.get('origin') != parsed.get('destination') and
        parsed.get('dest_IATA') and parsed.get('src_IATA')
    )
    print(bool(has_direct))
    return {
    "has_direct_flight": bool(has_direct)  # force bool!
    }

def llm_decider(state: State) -> State:
    from langchain_groq import ChatGroq
    llm=ChatGroq(model="llama3-70b-8192")
    
    tools = [direct_flights, via_nearby_airport, direct_trains, 
             multi_leg_train, train_partial_road, bus_or_cab_only]
    
    llm_with_tools = llm.bind_tools(tools)

    print("🚀 llm_decider() called with state keys:", list(state.keys()))
    """Main decision function to determine best travel route"""
    choice = state.get("human_review_result", "").lower()
    feedback = state.get("user_feedback", "No feedback given")

    print("CHOICE : ",choice)
    print("FEEDBACK : ",feedback)

    parsed = state.get("parsed", None)

    print(parsed)

    if choice == 'needs_modification':
        st.write(f"MODIFICATION REQUEST - Feedback: {feedback}")
        st.write(f"MODIFICATION REQUEST - Parsed: {parsed}")

    if not state.get("messages"):
        return {
            "messages": [AIMessage(content="❌ No messages found in state")],
            "decision": None
        }

    user_input = state["messages"][-1].content
    if not parsed:
        try:
            parsed = extract_trip_fields(user_input)
            print("LLM DECIDER : ",parsed)
        except Exception as e:
            return {
                "messages": [*state["messages"], AIMessage(content=f"❌ Failed to extract trip fields: {e}")],
                "decision": None
            }

    # Check direct flights only if not modifying
    if choice != 'needs_modification':
        direct_flights_check = find_direct_flights(parsed)

        if direct_flights_check["has_direct_flight"]:
            print("❌ ENTERED IF BLOCK - THIS SHOULD NOT HAPPEN!")
            return {
                "messages": [*state["messages"], AIMessage(content="✅ Direct flight available - using direct flight option")],
                "decision": "direct_flights",
                "parsed": parsed
            }

    # ✂️ Trim messages: only last 1 user + last 1 AI message (if any)
    recent_messages = []
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and not any(isinstance(x, HumanMessage) for x in recent_messages):
            recent_messages.insert(0, m)
        elif isinstance(m, AIMessage) and not any(isinstance(x, AIMessage) for x in recent_messages):
            recent_messages.insert(0, m)
        if len(recent_messages) == 2:
            break

    # Compose final prompt
    prompt = f"""
    USER FEEDBACK: {feedback}

    Trip plan: {parsed.get('origin')} to {parsed.get('destination')} on {parsed.get('start_date')} for {parsed.get('duration')} days.
    Nearby source: {parsed.get('nearby_src_city')} | Nearby destination: {parsed.get('nearby_dest_city')}

    Available travel options:
    1. direct_flights
    2. via_nearby_airport
    3. direct_trains
    4. multi_leg_train
    5. train_partial_road
    6. bus_or_cab_only

    Prioritize flights for very long distances unless feedback prefers trains or other modes.
    Return only ONE of the options above.Only Give the exact name of option as mentioned in available travel options. No explanation. Just return the name.
    If any feedback is given , give preference to that.
    """

    try:
        response = llm_with_tools.invoke([
            *recent_messages,
            HumanMessage(content=prompt)
        ])

        decision = response.content.strip().lower()
        
        print("\nDecision : ",decision)

        valid_options = {
            "direct_flights", "via_nearby_airport", "direct_trains",
            "multi_leg_train", "train_partial_road", "bus_or_cab_only"
        }

        if decision not in valid_options:
            print(f"❌ Invalid decision: '{decision}'")
            decision = "bus_or_cab_only"

        return {
            "messages": [*state["messages"], AIMessage(content=f"Selected route type: {decision}")],
            "decision": decision,
            "parsed": parsed
        }

    except Exception as e:
        print(f"❌ Exception in LLM decision: {e}")
        return {
            "messages": [*state["messages"], AIMessage(content=f"❌ Decision making failed: {e}")],
            "decision": "bus_or_cab_only"
        }

def extract_hub_city_names(response_text: str) -> list:
    """Extract city names from LLM response"""
    try:
        # Try JSON-style list extraction first
        city_list = re.findall(r'"([^"]*)"', response_text)
        if city_list:
            return [city.strip() for city in city_list if city.strip()]
        
        # Fallback to comma/line split
        city_list = [city.strip() for city in re.split(r'[\n,]', response_text) if city.strip()]
        return city_list[:3]  # Limit to 3 cities
    except Exception as e:
        print(f"Hub city extraction failed: {e}")
        return []

def multi_leg_train(state: State) -> State:
    """Get multi-leg train journey options via major hubs"""
    try:
        parsed = state["parsed"]
        
        # Get hub cities from LLM
        hub_prompt = f"""
Suggest 2-3 major railway junction cities for traveling from {parsed['origin']} to {parsed['destination']}.
Return format: ["HubCity1", "HubCity2", "HubCity3"]
No explanation needed.
"""
        
        hub_response = llm.invoke([*state["messages"], HumanMessage(content=hub_prompt)])
        hub_cities = extract_hub_city_names(hub_response.content)
        
        if not hub_cities:
            return {
                "messages": [*state["messages"], AIMessage(content="❌ Could not find suitable train hubs")],
                "decision": state.get("decision"),
                "parsed":parsed
            }
        
        date_ddmmyyyy = datetime.strptime(parsed["start_date"], "%Y-%m-%d").strftime("%d/%m/%Y")
        tool_outputs = []
        
        for hub in hub_cities:
            tool_outputs.append(f"--- 🛤️ Multi-Leg Train via **{hub}** ---")
            
            # First leg
            try:
                leg1 = get_etrain_trains(parsed["origin"], hub, date_ddmmyyyy)
                if isinstance(leg1, str) and "Empty" in leg1:
                    tool_outputs.append(f"🚆 **{parsed['origin']} → {hub}**: ❌ No trains found")
                else:
                    tool_outputs.append(f"🚆 **{parsed['origin']} → {hub}**:\n{leg1}")
            except Exception as e:
                tool_outputs.append(f"❌ Leg 1 error: {e}")
            
            # Second leg
            try:
                leg2 = get_etrain_trains(hub, parsed["destination"], date_ddmmyyyy)
                if isinstance(leg2, str) and "Empty" in leg2:
                    tool_outputs.append(f"🚆 **{hub} → {parsed['destination']}**: ❌ No trains found")
                else:
                    tool_outputs.append(f"🚆 **{hub} → {parsed['destination']}**:\n{leg2}")
            except Exception as e:
                tool_outputs.append(f"❌ Leg 2 error: {e}")
        
        combined_output = "\n\n".join(tool_outputs)
        return {
            "messages": [*state["messages"], AIMessage(content=combined_output)],
            "decision": state.get("decision"),
            "parsed":parsed
        }
        
    except Exception as e:
        error_msg = f"❌ Multi-leg Train Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision"),
            "parsed":parsed
        }

def train_partial_road(state: State) -> State:
    """Get mixed train and road journey options"""
    try:
        parsed = state["parsed"]
        
        # Get hub cities from LLM
        hub_prompt = f"""
Suggest 2-3 good intermediate cities for mixed train/road travel from {parsed['origin']} to {parsed['destination']}.
Return format: ["HubCity1", "HubCity2"]
No explanation needed.
"""
        
        hub_response = llm.invoke([*state["messages"], HumanMessage(content=hub_prompt)])
        hub_cities = extract_hub_city_names(hub_response.content)
        
        if not hub_cities:
            return {
                "messages": [*state["messages"], AIMessage(content="❌ Could not find suitable intermediate cities")],
                "decision": state.get("decision") 
            }
        
        all_results = []
        for hub in hub_cities:
            try:
                journey = dynamic_breakup_journey(parsed, hub)
                all_results.append(f"--- 🧭 Mixed Journey via {hub} ---\n{journey}")
            except Exception as e:
                all_results.append(f"❌ Could not process journey via {hub}: {e}")
        
        combined_output = "\n\n".join(all_results)
        return {
            "messages": [*state["messages"], AIMessage(content=combined_output)],
            "decision": state.get("decision") 
        }
        
    except Exception as e:
        error_msg = f"❌ Train Partial Road Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision"),
            "parsed":parsed
        }

def bus_or_cab_only(state: State) -> State:
    """Get bus and cab only travel options"""
    try:
        parsed = state["parsed"]
        origin = parsed["origin"]
        dest = parsed["destination"]
        date = parsed["start_date"]
        date_ddmmyyyy = datetime.strptime(date, "%Y-%m-%d").strftime("%d/%m/%Y")
        
        results = []
        
        # Try direct bus
        try:
            direct_bus = get_abhibus_buses(origin, dest, date_ddmmyyyy)
            has_direct_bus = bool(direct_bus and str(direct_bus).strip())
        except Exception as e:
            direct_bus = None
            has_direct_bus = False
            bus_error = f"❌ Bus Error: {e}"
        
        # Check if cab is practical via LLM
        try:
            cab_prompt = f"""
Is a cab practical for travel from {origin} to {dest}? Consider distance and cost.
Reply only 'YES' or 'NO'.
"""
            cab_response = llm.invoke([*state["messages"], HumanMessage(content=cab_prompt)])
            cab_allowed = cab_response.content.strip().upper() == "YES"
        except Exception as e:
            cab_allowed = False
            cab_error = f"❌ Cab decision error: {e}"
        
        # Get cab options if allowed
        direct_cab = None
        if cab_allowed:
            try:
                direct_cab = find_cabs(origin, dest, date,parsed['Persons'])
                has_direct_cab = bool(direct_cab and str(direct_cab).strip())
            except Exception as e:
                direct_cab = None
                has_direct_cab = False
                cab_error = f"❌ Cab Error: {e}"
        else:
            has_direct_cab = False
            cab_error = "🚫 Cab judged impractical for this route"
        
        # Show results
        if has_direct_bus or has_direct_cab:
            results.append(f"### 🚐 Direct Journey: {origin} → {dest}")
            if has_direct_bus:
                results.append(f"🚌 **Buses**:\n{direct_bus}")
            else:
                results.append(f"🚌 **Buses**: {bus_error}")
            
            if has_direct_cab:
                results.append(f"🚖 **Cabs**:\n{direct_cab}")
            else:
                results.append(f"🚖 **Cabs**: {cab_error}")
        else:
            # Try breakup journey
            results.append(f"⚠️ No direct options found. Trying breakup journeys...")
            
            hub_prompt = f"""
Suggest 2 intermediate cities for bus/cab travel from {origin} to {dest}.
Return format: ["HubCity1", "HubCity2"]
No explanation needed.
"""
            
            hub_response = llm.invoke([*state["messages"], HumanMessage(content=hub_prompt)])
            hub_cities = extract_hub_city_names(hub_response.content)
            
            if not hub_cities:
                results.append("❌ Could not find suitable intermediate cities")
            else:
                for hub in hub_cities:
                    try:
                        journey = dynamic_breakup_journey_cab(parsed, hub)
                        results.append(f"\n--- 🧭 Journey via {hub} ---\n{journey}")
                    except Exception as e:
                        results.append(f"❌ Could not process journey via {hub}: {e}")
        
        combined_output = "\n\n".join(results)
        return {
            "messages": [*state["messages"], AIMessage(content=combined_output)],
            "decision": state.get("decision") 
        }
        
    except Exception as e:
        error_msg = f"❌ Bus/Cab Only Tool failed: {e}"
        return {
            "messages": [*state["messages"], AIMessage(content=error_msg)],
            "decision": state.get("decision") 
        }

def dynamic_breakup_journey(parsed: dict, hub_city: str) -> str:
    """Build mixed train/road journey options"""
    try:
        date_ddmmyyyy = datetime.strptime(parsed["start_date"], "%Y-%m-%d").strftime("%d/%m/%Y")
        origin = parsed["origin"]
        dest = parsed["destination"]
        date = parsed["start_date"]
        
        results = []
        
        # Option 1: Train first, then road
        results.append(f"### 🅰️ Train first ({origin} → {hub_city}), then road to {dest}")
        
        try:
            train1 = get_etrain_trains(origin, hub_city, date_ddmmyyyy)
            results.append(f"🚆 **{origin} → {hub_city}**: {train1}")
        except Exception as e:
            results.append(f"❌ Train error: {e}")
        
        try:
            buses = get_abhibus_buses(hub_city, dest, date_ddmmyyyy)
            cabs = find_cabs(hub_city, dest, date,parsed["Persons"])
            results.append(f"🚌 **Buses {hub_city} → {dest}**: {buses}")
            results.append(f"🚖 **Cabs {hub_city} → {dest}**: {cabs}")
        except Exception as e:
            results.append(f"❌ Road options error: {e}")
        
        # Option 2: Road first, then train
        results.append(f"\n### 🅱️ Road first ({origin} → {hub_city}), then train to {dest}")
        
        try:
            buses2 = get_abhibus_buses(origin, hub_city, date_ddmmyyyy)
            cabs2 = find_cabs(origin, hub_city, date,parsed["Persons"])
            results.append(f"🚌 **Buses {origin} → {hub_city}**: {buses2}")
            results.append(f"🚖 **Cabs {origin} → {hub_city}**: {cabs2}")
        except Exception as e:
            results.append(f"❌ Road options error: {e}")
        
        try:
            train2 = get_etrain_trains(hub_city, dest, date_ddmmyyyy)
            results.append(f"🚆 **{hub_city} → {dest}**: {train2}")
        except Exception as e:
            results.append(f"❌ Train error: {e}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"❌ Journey planning error: {e}"

def dynamic_breakup_journey_cab(parsed: dict, hub_city: str) -> str:
    """Build all bus/cab combination journeys"""
    try:
        origin = parsed["origin"]
        dest = parsed["destination"]
        date = parsed["start_date"]
        date_ddmmyyyy = datetime.strptime(date, "%Y-%m-%d").strftime("%d/%m/%Y")
        
        results = []
        combinations = [
            ("🚌 Bus", "🚌 Bus", get_abhibus_buses, get_abhibus_buses),
            ("🚌 Bus", "🚖 Cab", get_abhibus_buses, find_cabs),
            ("🚖 Cab", "🚌 Bus", find_cabs, get_abhibus_buses),
            ("🚖 Cab", "🚖 Cab", find_cabs, find_cabs)
        ]
        
        for i, (mode1, mode2, func1, func2) in enumerate(combinations, 1):
            results.append(f"### Option {i}: {mode1} {origin} → {hub_city}, {mode2} {hub_city} → {dest}")
            
            try:
                if func1 == find_cabs:
                    leg1 = func1(origin, hub_city, date)
                else:
                    leg1 = func1(origin, hub_city, date_ddmmyyyy)
                results.append(f"**Leg 1**: {leg1}")
            except Exception as e:
                results.append(f"❌ Leg 1 error: {e}")
            
            try:
                if func2 == find_cabs:
                    leg2 = func2(hub_city, dest, date)
                else:
                    leg2 = func2(hub_city, dest, date_ddmmyyyy)
                results.append(f"**Leg 2**: {leg2}")
            except Exception as e:
                results.append(f"❌ Leg 2 error: {e}")
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"❌ Journey planning error: {e}"

def llm_transportation_combiner(state: State) -> State:
    from langchain_groq import ChatGroq
    llm=ChatGroq(model="llama3-70b-8192")
    """Summarize and recommend best transportation options"""
    try:
        full_conversation = state["messages"]
        
        if not full_conversation:
            return {
                **state,  # Include all existing state
                "messages": [AIMessage(content="❌ No transportation data to summarize")],
                "decision": state.get("decision"),
                "arrival_time": None,
                "transportation_summary": "No transportation data to summarize"
            }
        
        # Get the last tool output
        tool_outputs = full_conversation[-1].content
        feedback = state.get("user_feedback", "").strip()
        parsed = state.get("parsed", {})
        
        prompt = f"""
Based on the transportation search results below, provide a comprehensive travel summary:

SEARCH RESULTS:
{tool_outputs}

TRIP DETAILS:
- Origin: {parsed.get('origin', 'Unknown')}
- Destination: {parsed.get('destination', 'Unknown')}
- Travel Date: {parsed.get('start_date', 'Unknown')}
- Duration: {parsed.get('duration', 'Unknown')} days

USER FEEDBACK(Optional):
{feedback if feedback else 'No feedback provided.'}

Please provide:

1. **Travel Summary**: Brief overview of available options.

2. **Detailed Options**: For each viable route, include:
   - Mode of transport
   - Departure time and travel duration
   - **Estimated Arrival Time at Destination** (format as HH:MM if possible)
   - Approximate cost
   - Booking suggestions or URLs (if any)

3. **Recommendation**: Suggest the best option considering travel time, arrival time, cost, and convenience.

4. **Alternative Option**: A second-best fallback if the preferred option isn't available.

5. **ESTIMATED_ARRIVAL_TIME**: Provide the estimated arrival time to the final destination for the recommended option in HH:MM format (e.g., "14:30").Add some delays in the time .
    Just give me the answer in format : Arrival_time : HH:MM format

6. Provide the estimated budget for persons in full travel from Source to Destination. Consider all expenses involved in travel .Show in format:
    Estimated Budget of Travel : estimated_budget

Format clearly in sections. Ensure arrival time is explicitly estimated where possible.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract arrival time from the response
        arrival_time = extract_arrival_time(response.content)
        
        # Remove this line - it's redundant
        # state["transportation_summary"]=response.content
        
        return {
            **state,  # Include all existing state fields
            "messages": [*state["messages"], response],
            "decision": state.get("decision"),
            "arrival_time": arrival_time,
            "transportation_summary": response.content,  # This will now persist
        }
        
    except Exception as e:
        error_msg = f"❌ Transportation summary failed: {e}"
        return {
            **state,  # Include all existing state
            "messages": [AIMessage(content=error_msg)],
            "decision": state.get("decision"),
            "arrival_time": None,
            "transportation_summary": error_msg
        }

def extract_arrival_time(response_text: str) -> str:
    """Extract arrival time from LLM response"""
    try:
        # Look for ESTIMATED_ARRIVAL_TIME section
        arrival_match = re.search(r'ESTIMATED_ARRIVAL_TIME[:\s]*([0-9]{1,2}:[0-9]{2})', response_text, re.IGNORECASE)
        if arrival_match:
            return arrival_match.group(1)
        
        # Look for arrival time patterns in the text
        time_patterns = [
            r'arrival time[:\s]*([0-9]{1,2}:[0-9]{2})',
            r'arrives at[:\s]*([0-9]{1,2}:[0-9]{2})',
            r'reaching at[:\s]*([0-9]{1,2}:[0-9]{2})',
            r'will arrive[:\s]*([0-9]{1,2}:[0-9]{2})',
            r'estimated arrival[:\s]*([0-9]{1,2}:[0-9]{2})'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no specific arrival time found, return None
        return None
        
    except Exception as e:
        print(f"Arrival time extraction failed: {e}")
        return None

def should_continue(state: State) -> str:
    """Determine which tool to call based on decision"""
    decision = state.get("decision")
    if not decision:
        return "llm_transportation_combiner"
    
    # Map decisions to tool names
    tool_mapping = {
        "direct_flights": "direct_flights",
        "via_nearby_airport": "via_nearby_airport", 
        "direct_trains": "direct_trains",
        "multi_leg_train": "multi_leg_train",
        "train_partial_road": "train_partial_road",
        "bus_or_cab_only": "bus_or_cab_only"
    }
    
    return tool_mapping.get(decision, "bus_or_cab_only")

# %%
def human_node(state: State) -> State:
    st.subheader("👤 Human Review of Transportation Options")

    # Show current transport summary
    transport_summary = state.get("transportation_summary", "No options available")
    st.markdown(f"""
    <div class="transport-card">
        {transport_summary}
    </div>
    """, unsafe_allow_html=True)

    # Get human choice through select UI
    review_choice = st.selectbox("What would you like to do?", [
        "Choose an action", "Approve", "Modify", "Continue with current options"
    ], key="review_choice_select")

    # Optional feedback input if Modify selected
    feedback = ""
    if review_choice == "Modify":
        feedback = st.text_area(
            "📝 Please describe what to modify:",
            placeholder="e.g., prefer train, avoid expensive options",
            key="user_feedback_input"
        )

    if st.button("Submit Decision", key="submit_decision_btn"):
        if review_choice == "Approve":
            state["human_review_result"] = "approved"
            st.success("✓ Transportation options approved!")

        elif review_choice == "Modify":
            if not feedback.strip():
                st.warning("⚠️ Please enter feedback before submitting.")
            else:
                state["user_feedback"] = feedback
                state["human_review_result"] = "needs_modification"
                st.info("↻ Requesting modifications...")

        elif review_choice == "Continue with current options":
            state["human_review_result"] = "approved"
            st.success("✓ Continuing with current options...")

        else:
            st.warning("❗ Please select an action before submitting.")

    # Show feedback if already in state
    if state.get("human_review_result") == "needs_modification" and state.get("user_feedback"):
        st.markdown(f"""
        <div style="background-color: black;padding:10px;border-radius:10px;border-left:5px solid red;">
            <strong>🔁 Feedback submitted:</strong><br>
            {state['user_feedback']}
        </div>
        """, unsafe_allow_html=True)

    return state

def llm_feedback_router(state: State) -> State:
    from langchain_groq import ChatGroq
    llm=ChatGroq(model="llama3-70b-8192")
    """Handle user feedback and route to appropriate node"""
    plan = state.get("plan", "⚠️ No plan available.")
    print("\n🧭 Here's your current day-wise itinerary:\n")
    print(plan)
    
    user_input = input("Would you like to customize the plan? Say 'Yes' or 'No': ")
    
    if user_input.lower() in ["yes", "y"]:
        user_input2 = input("What customizations would you like to do? ")
        state['human_feedback'] = user_input2
        
        feedback = state.get("human_feedback", "")
        
        prompt = f"""
You are a travel assistant. A user has given the following feedback on their travel plan:

"{feedback}"

Decide which part needs to be updated:
- "Hotels" if they want to change hotels or accommodation
- "DayWise_Plan" if they want to change the itinerary, activities, or schedule
- "llm_decider" if they want to change transportation options
- "END" if no change is required or feedback is unclear

Just respond with the name of the component. Don't include anything else.
"""
        
        result = llm.invoke([HumanMessage(content=prompt)])
        decision = result.content.strip()
        
        return {
            **state,
            "llm_routing_decision": decision
        }
    else:
        return {
            **state,
            "llm_routing_decision": "END"
        }

# %%
# Build the graph
from langchain_core.runnables import Runnable

def build_trip_graph() -> Runnable:
    from langchain_groq import ChatGroq
    llm=ChatGroq(model="llama3-70b-8192")
    memory = MemorySaver()
    graph_builder = StateGraph(State)

    tools = [direct_flights, via_nearby_airport, direct_trains, 
            multi_leg_train, train_partial_road, bus_or_cab_only]
    
    # llm_with_tools = llm.bind_tools(tools)

    graph_builder.add_node("llm_decider",llm_decider)
    for tool in tools:
        graph_builder.add_node(tool.__name__, tool)

    def llm_feedback_router(state: State) -> State:
        plan = state.get("plan", "⚠️ No plan available.")
        print("\n🧭 Here's your current day-wise itinerary:\n")
        print(plan)
        
        user_input = input("Would you like to customize the plan? Say 'Yes' or 'No': ")
        
        if user_input.lower() in ["yes", "y"]:
            user_input2 = input("What customizations would you like to do? ")
            state['human_feedback'] = user_input2
            
            feedback = state.get("human_feedback", "")
            prompt = f"""
You are a travel assistant. A user has given the following feedback on their travel plan:

"{feedback}"

Decide which part needs to be updated:
- "Hotels" if they want to change hotels or accommodation
- "DayWise_Plan" if they want to change the itinerary, activities, or schedule
- "llm_decider" if they want to change transportation options
- "END" if no change is required or feedback is unclear

Just respond with the name of the component. Don't include anything else.
"""
            result = llm.invoke([HumanMessage(content=prompt)])
            decision = result.content.strip()
            
            return {
                **state,
                "llm_routing_decision": decision
            }
        else:
            return {
                **state,
                "llm_routing_decision": "END"
            }


    graph_builder.add_node("llm_transportation_combiner", llm_transportation_combiner)
    graph_builder.add_node("Weather_Forecast", get_weather_forecast_node)
    graph_builder.add_node("Human_Review", human_node)
    graph_builder.add_node("DayWise_Plan", wikivoyage_daywise_plan)
    graph_builder.add_node("Hotels", search_hotels_serpapi)
    graph_builder.add_node("User_Feedback", llm_feedback_router)

    graph_builder.set_entry_point("llm_decider")

    graph_builder.add_conditional_edges(
        "llm_decider",
        should_continue,
        {
            "direct_flights": "direct_flights",
            "via_nearby_airport": "via_nearby_airport",
            "direct_trains": "direct_trains", 
            "multi_leg_train": "multi_leg_train",
            "train_partial_road": "train_partial_road",
            "bus_or_cab_only": "bus_or_cab_only",
            "llm_transportation_combiner": "llm_transportation_combiner"
        }
    )

    for tool in tools:
        graph_builder.add_edge(tool.__name__, "llm_transportation_combiner")

    graph_builder.add_edge("llm_transportation_combiner", "Human_Review")

    def human_review_condition(state):
        review_result = state.get("human_review_result", "approved")
        if review_result == "approved":
            return "Weather_Forecast"
        elif review_result == "needs_modification":
            return "llm_decider"
        else:
            return "Weather_Forecast"

    graph_builder.add_conditional_edges(
        "Human_Review",
        human_review_condition,
        {
            "Weather_Forecast": "Weather_Forecast",
            "llm_decider": "llm_decider"
        }
    )

    graph_builder.add_edge("Weather_Forecast", "Hotels")
    graph_builder.add_edge("Hotels", "DayWise_Plan")
    graph_builder.add_edge("DayWise_Plan", "User_Feedback")

    def route_from_llm_decision(state: State) -> str:
        return state.get("llm_routing_decision", "END")

    graph_builder.add_conditional_edges(
        "User_Feedback",
        route_from_llm_decision,
        {
            "Hotels": "Hotels",
            "DayWise_Plan": "DayWise_Plan", 
            "llm_decider": "llm_decider",
            "END": END
        }
    )

    graph = graph_builder.compile(checkpointer=memory)
    return graph

import streamlit as st
def plan_my_trip(origin, destination, start_date,end_date, num_travelers, interests,graph):
    from langchain.schema import HumanMessage
    from uuid import uuid4
    import streamlit as st
    st.write("✅ plan_my_trip() called")
    user_message = (
        f"Plan a trip from {origin} to {destination} from {start_date.strftime('%d %B')} {datetime.now().year} to {end_date.strftime('%d %B')}{datetime.now().year}. We are {num_travelers} persons. Interests : {interests}"
    )

    config = {"configurable": {"thread_id": str(uuid4())}}
    input_state = {
        "messages": [HumanMessage(content=user_message)]
    }

    # This assumes your `graph` is already initialized
    output = graph.invoke(input_state, config=config)
    st.session_state["state"] = output
    return output
from amadeus import Client, ResponseError
from datetime import datetime, timedelta

# Initialize Amadeus client
amadeus = Client(
    client_id='oAzJgebAHH4JxgZUDeadSA2Ks8J6kLPj',
    client_secret='7DZWXlP8C2MZtP2U'
)

AFFILIATE_MARKER = "641880"
airport_name_cache = {}

def get_airport_name(iata_code):
    if iata_code in airport_name_cache:
        return airport_name_cache[iata_code]
    try:
        response = amadeus.reference_data.locations.get(
            keyword=iata_code,
            subType='AIRPORT'
        )
        for item in response.data:
            if item["iataCode"] == iata_code:
                name = item["name"]
                airport_name_cache[iata_code] = name
                return name
    except ResponseError:
        pass
    return f"{iata_code} Airport"

def parse_iso_time(iso_str):
    return datetime.fromisoformat(iso_str)

def calculate_layovers(segments):
    layover_time = timedelta()
    layover_cities = []
    for i in range(1, len(segments)):
        prev_arrival = parse_iso_time(segments[i - 1]['arrival']['at'])
        curr_departure = parse_iso_time(segments[i]['departure']['at'])
        diff = curr_departure - prev_arrival
        layover_time += diff
        layover_city = segments[i]['departure']['iataCode']
        layover_cities.append(get_airport_name(layover_city))
    return layover_time, layover_cities

# ✅ LangGraph Tool-compatible function
def search_flights_tool(origin: str, destination: str, departure_date: str, return_date: str,src_IATA:str,dest_IATA:str, adults: int = 1, currency: str = "INR", max_results: int = 10) -> dict:
    """
    Searches flight deals using this tool using amadeus API
    Args:str
    origin:str
    destination:str
    departure_date:str format:  YYYY-MM-DD
    return_date:str    format:  YYYY-MM-DD
    src_IATA:str (If we are not able to get IATA code for origin , then we will use src_IATA)
    dest_IATA:str(If we are not able to get IATA code for destination , then we will use dest_IATA)
    adults: int
    currency:str
    max_results:int
    Returns

    dict
    """
        
    results = []
    # try:
    response = amadeus.shopping.flight_offers_search.get(
        originLocationCode=src_IATA.upper(),
        destinationLocationCode=dest_IATA.upper(),
        departureDate=departure_date,
        returnDate=return_date,
        adults=adults,
        currencyCode=currency,
        max=max_results
    )
    print(response)

    for offer in response.data:
        price = offer['price']['total']
        itinerary = offer['itineraries'][0]
        segments = itinerary['segments']

        try:
            origin_code = segments[0]['departure']['iataCode']
            destination_code = segments[-1]['arrival']['iataCode']
        except ResponseError as error:
            print("Error code",str(error))
        print(origin_code,destination_code)
        origin_name = get_airport_name(origin_code)
        destination_name = get_airport_name(destination_code)

        departure = segments[0]['departure']['at']
        returning = offer['itineraries'][-1]['segments'][-1]['arrival']['at']
        stops = len(segments) - 1

        layover_time, layover_cities = calculate_layovers(segments)

        dep_date_obj = datetime.fromisoformat(departure)
        ddmm = dep_date_obj.strftime("%d%m")
        booking_link = f"https://www.aviasales.com/search/{origin_code}{ddmm}{destination_code}1?marker={AFFILIATE_MARKER}"

        results.append({
            "flight": f"{origin_name} ({origin_code}) → {destination_name} ({destination_code})",
            "departure": departure,
            "return": returning,
            "price": f"₹{price}",
            "stops": stops,
            "layover_time": str(layover_time) if stops > 0 else "Direct",
            "layover_cities": layover_cities if stops > 0 else [],
            "booking_link": booking_link
        })

    return {"flights": results}

    # except ResponseError as error:
    #     return {"error": str(error)}


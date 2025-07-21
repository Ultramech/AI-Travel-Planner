import requests

SERPAPI_KEY = "ffae7c6f5a305447f1471cea40a87a6df688bf286ce29c8a3df10a48c3955677"  # 🔑 Replace with your actual key

def search_hotels_serpapi2(city: str, checkin: str, checkout: str,duration:int, feedback : str,persons:int=1) -> str:
    
    cleaned_feedback = feedback.lower().replace("feedback", "").strip()

    params = {
        "engine": "google_hotels",
        "q": f"hotels in {city} {cleaned_feedback} ",
        "check_in_date": checkin,
        "check_out_date": checkout,
        "adults":persons,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "sort_by":0,
        "api_key": SERPAPI_KEY
    }
    resp = requests.get("https://serpapi.com/search", params=params)
    if resp.status_code != 200:
        return f"❌ Error: HTTP {resp.status_code}"
    data = resp.json()
    hotels = data.get("properties") or data.get("hotels")
    if not hotels:
        return "🚫 No hotels found—Google may not cover this area."
    top = hotels[:7]
    out = f"🏨 Top {len(top)} Hotels in {city} from {checkin} to {checkout}:\n\n"
    for h in top:
        name = h.get("name", "Unknown")
        price = h.get("rate_per_night", {}).get("lowest", "N/A")
        rating = h.get("overall_rating", h.get("rating", "N/A"))
        link = h.get("link", "")
        total_price=price*duration
        out += f"🔹 **{name}**\n💰 {total_price} for {persons} persons | ⭐ {rating}\n🔗 {link}\n\n"
    return out
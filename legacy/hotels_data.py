def search_hotels_serpapi2(city: str, checkin: str, checkout: str, duration: int, feedback: str, persons: int = 1) -> str:
    import requests
    SERPAPI_KEY = "<REDACTED — a live key was hardcoded here; see README issue #39>"
    cleaned_feedback = feedback.lower().replace("feedback", "").strip()

    params = {
        "engine": "google_hotels",
        "q": f"hotels in {city}",
        "check_in_date": checkin,
        "check_out_date": checkout,
        "adults": persons,
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    if cleaned_feedback:
        params["q"] += f" {cleaned_feedback}"

    resp = requests.get("https://serpapi.com/search", params=params)
    print("🔍 SerpAPI Request URL:", resp.url)  # for debugging

    if resp.status_code != 200:
        return f"❌ Error: HTTP {resp.status_code} - {resp.text}"

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

        try:
            total_price = float(price) * duration if price != "N/A" else "N/A"
        except:
            total_price = "N/A"

        out += f"🔹 **{name}**\n💰 {total_price} for {persons} persons | ⭐ {rating}\n🔗 {link}\n\n"

    return out

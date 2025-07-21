def generate_daywise_plan(query: str, destination:str,interest:str,weather_data:list,duration:str,arrival_time:str,arrival_date:str,hotel_info:str,persons:int,feedback:str) -> tuple:

    from langchain_community.document_loaders import WebBaseLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import Chroma

    # destination = parsed["destination"]
    country = "India"
    # interest = parsed.get("preferences", "sightseeing")
    # weather_data = parsed.get("weather_data", [])
    # duration = parsed.get("duration", "3")  # Default to 3 days if not specified
    # arrival_time=parsed.get("arrival_time","14:00")
    # arrival_date=parsed.get("start_date")
    # Format weather context for the LLM
    weather_context = []
    for day in weather_data:
        weather_context.append(
            f"{day['day_name']}: {day['condition']} ({day['min_temp']}°C to {day['max_temp']}°C), "
            f"Rain: {day['rain_chance']}%, Humidity: {day['humidity']}%"
        )
    weather_summary = "\n".join(weather_context)

    url1 = f"https://en.wikivoyage.org/wiki/{destination}_({country})"
    url2 = f"https://en.wikivoyage.org/wiki/{destination}"

    # Load pages with fallback
    try:
        doc1 = WebBaseLoader(url1).load()
    except Exception:
        doc1 = []
    try:
        doc2 = WebBaseLoader(url2).load()
    except Exception:
        doc2 = []

    docs = doc1 + doc2
    if not docs:
        return f"🚫 Could not find WikiVoyage info for {destination}."

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents, embeddings)

    # Retrieval + LLM
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    print("Persons passed in generate_daywise_plan : ",{persons})
    
    # hotel_price=hotel_info["total_price"]

    # print(hotel_info["total_price"])
# LLM query with hotel info injected
    full_query = (
        f"Create a {duration}-day itinerary for {destination} with these special requirements:\n"
        f"- Arrival on {arrival_date} at {arrival_time}\n"
        f"- User interests: {interest}\n"
        f"- Weather forecast:\n{weather_summary}\n"
        f"Hotels_info : {hotel_info}"

        f"USER FEEDBACK: {feedback}\n" 
        "SPECIFIC INSTRUCTIONS:\n"
        "1. For ARRIVAL DAY:\n"
        f"   - Plan only activities after {arrival_time}\n"
        "   - Suggest nearby dinner options\n"
        "   - Include light activities suitable for arrival day\n"
        "   - Recommend check-in friendly locations\n"
        "   - Suggest something near or within easy reach of the hotel\n\n"
        "2. For subsequent days:\n"
        "   - Full day plans from morning to night\n"
        "   - Group nearby attractions by transport\n"
        "   - Include weather-appropriate activities\n"
        "   - Add travel time estimates between locations\n"
        "   - Start from hotel and return to hotel each day\n\n"


        "Provide me the estimated budget of full trip incluing hotel .Also add travelling and food and any miscellaneous expense expenses in hotel_price." 
        "Provide the estimated budget in breakdown , and at last in format:"
        "Estimated_budget:estimated budget" 
        
        "Format with clear day headers and time-based schedules."
    )
    
    response = qa_chain.invoke(full_query)
    # budget = extract_estimated_budget_from_plan(response["result"])
    itinerary = f"🧭 **{duration}-Day Itinerary for {destination}**\n\n" + response["result"]
    return itinerary

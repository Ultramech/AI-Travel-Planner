

def find_cabs(origin: str, destination: str, date: str = None, time: str = None, persons:int=1) -> dict:
    from langchain_tavily import TavilySearch

    tool = TavilySearch()
    
    prompt = f"Give me cab details from {origin} to {destination}"
    if persons:
        prompt+= f"for {persons} persons"
    if date:
        prompt += f" on {date}"
    if time:
        prompt += f" at {time}"
    prompt += ". Provide the fare also."

    return tool.invoke(prompt)

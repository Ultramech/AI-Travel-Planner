from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_groq import ChatGroq
from langchain.schema.messages import HumanMessage
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
import pandas as pd

def extract_trip_fields(user_input:str):

    # 1. Define the fields to extract
    response_schemas = [
        ResponseSchema(name="origin", description="Starting location (e.g., Delhi)"),
        ResponseSchema(name="destination", description="Trip destination (e.g., Paris)"),
        ResponseSchema(name="start_date", description="Start date of trip (YYYY-MM-DD)"),
        ResponseSchema(name="end_date", description="End date of trip (YYYY-MM-DD)"),
        ResponseSchema(name="duration", description="Duration of trip (in days)"),
        # ResponseSchema(name="budget", description="Approximate budget in INR"),
        ResponseSchema(name="preferences", description="List of interests like food, nature, adventure"),
        ResponseSchema(name="src_IATA", description="Origin IATA code"),
        ResponseSchema(name="dest_IATA", description="Destination IATA code"),
        ResponseSchema(name="nearby_src_city", description="Source IATA code city "),
        ResponseSchema(name="nearby_dest_city", description="Destination IATA code city "),
        ResponseSchema(name="Persons",description="No. of persons going to the trip")
    ]

    # 2. Create the parser
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    # 3. Create the prompt

    prompt = PromptTemplate(
        template="""
        Extract the following information from the text below.
        If u get the start date and duration , automatically calculate end date accordingly
        {format_instructions}

        Text:
        {input_text}
        """,
        input_variables=["input_text"],
        partial_variables={"format_instructions": format_instructions}
    )

    # 4. Input text
    system_instructions = """
    You are a helpful travel planner.
    ONLY return valid JSON with the following keys:
    "origin", "destination", "start_date", "end_date", "duration", "budget", "preferences", "src_IATA", "dest_IATA","nearby_src_city","nearby_dest_city","Persons"
    Do NOT include text or explanations outside the JSON block.
    """
    
    user_input+='Also convert the source and destination into IATA codes.If the IATA code does not exists tell, then tell nearby city IATA code ,Also tell the nearby city name whose IATA Code you are using ,  for eg.) Aligarh has no IATA code , so tell IATA code for Delhi as DEL and also telll the city name, Also tell the IATA code of destination , and if IATA code of destination is not available , tell nearby city IATA Code and also mention the city name'
    'In neaby_dest_city tell the name of city whose IATA code you are using. ALways make sure that src_IATA, dest_IATA, nearby_src_city,nearby_dest_city are never empty'
    
    from langchain.schema import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=prompt.format(input_text=user_input))
    ]
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    response = llm.invoke(messages)
    
    print("Raw LLM Output:\n", response.content)

    # 8. Parse the structured output
    parsed = output_parser.parse(response.content)

    # parsed["src_IATA"] = get_iata_code(parsed.get("origin", ""))
    # parsed["dest_IATA"] = get_iata_code(parsed.get("destination", ""))

    return parsed
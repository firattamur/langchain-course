from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def run_with_no_tools(prompt: str) -> str:
    prompt_template = PromptTemplate.from_template(
        "Sen bir yapay zeka asistanısın. Soruyu cevapla: {question}"
    )
    llm = ChatOllama(model="qwen3:1.7b", temperature=0)
    output_parser = StrOutputParser()

    chain = (
        prompt_template
        | llm
        | output_parser
    )

    response = chain.invoke({"question": prompt})

    return response

@tool
def get_current_weather() -> dict:
    """
    Get the current weather for the user's location.
    """
    ip_response = requests.get("https://ipinfo.io/json")
    ip_data = ip_response.json()

    latitude, longitude = ip_data["loc"].split(",")

    weather_response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    )

    weather_data = weather_response.json()
    return weather_data["current_weather"]


def run_with_tools(prompt: str) -> str:
    prompt_template = PromptTemplate.from_template(
        "Sen bir yapay zeka asistanısın. Soruyu cevapla: {question}"
    )
    llm = ChatOllama(model="qwen3:1.7b", temperature=0)
    llm_with_tools = llm.bind_tools([get_current_weather])

    chain = (
        prompt_template
        | llm_with_tools
    )

    response = chain.invoke({"question": prompt})

    if hasattr(response, "tool_calls"):
        print(f"Tool calls made: {response.tool_calls}")
        tool_call = response.tool_calls[0]
        
        if tool_call['name'] == "get_current_weather":
            current_weather = get_current_weather.invoke({})

            messages = [
                HumanMessage(content=prompt),
                AIMessage(content=response.content, tool_calls=[tool_call]),
                ToolMessage(content=f"Current weather: {current_weather}", tool_call_id=tool_call['id'])
            ]

            response = llm_with_tools.invoke(messages)
            print(f"Final response after tool call: {response.content}")

    else:
        print("No tool calls made.")

    return response


if __name__ == "__main__":
    prompt: str = "Şu an hava kaç derece?"

    # response_llm_with_no_tools = run_with_no_tools(prompt)
    # print(f"Response without tools: {response_llm_with_no_tools}")

    response_llm_with_tools = run_with_tools(prompt)
    print(f"Response with tools: {response_llm_with_tools}")



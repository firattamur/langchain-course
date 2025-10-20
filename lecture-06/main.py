import json
from typing import Literal, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

tools: list = [add, multiply]
tools_by_name = {tool.name: tool for tool in tools}

model: ChatOllama = ChatOllama(
    model="qwen3:1.7b"
)
model_with_tools: ChatOllama = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_tool_node(state: AgentState) -> AgentState: 
    outputs: list[ToolMessage] = []

    messages: list[BaseMessage] = state["messages"]
    if not messages:
        return {"messages": outputs}
    
    last_message: BaseMessage = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        
        if not tool:
            raise ValueError(f"Tool {tool_call['name']} not found.")
        
        tool_args = tool_call["args"]
        tool_result = tool.invoke(tool_args)

        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": outputs}


def call_model_node(state: AgentState) -> AgentState:
    system_message: SystemMessage = SystemMessage(
        content="You are a helpful assistant that can use tools to perform calculations."
    )

    messages = state["messages"]
    response: AIMessage = model_with_tools.invoke(
        [system_message] + messages
    )

    return {"messages": [response]}


def should_continue_edge(state: AgentState) -> Literal["continue", "end"]:
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_message: BaseMessage = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if tool_calls:
        return "continue"
    
    return "end"


# create graph builder
graph_builder = StateGraph(AgentState)


# add nodes
graph_builder.add_node("model", call_model_node)
graph_builder.add_node("tools", call_tool_node)


# add edges
graph_builder.add_edge(START, "model")
graph_builder.add_edge("tools", "model")
graph_builder.add_conditional_edges(
    "model",
    should_continue_edge,
    {
        "continue": "tools",
        "end": END
    }
)


# compile graph
graph = graph_builder.compile()

# get grap image
graph_image: bytes = graph.get_graph().draw_mermaid_png() 

# save the graph image
with open("graph.png", "wb") as f:
    f.write(graph_image)


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    inputs: AgentState = {"messages": [HumanMessage(content="What is 5 plus 3 multiplied by 2?")]}
    print_stream(graph.stream(inputs, stream_mode="values"))

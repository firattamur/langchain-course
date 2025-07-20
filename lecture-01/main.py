from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def main():
    prompt_template = PromptTemplate(
        input_variables=["name"],
        template="Günaydin {name}, nasılsın?",
    )

    llm_openai = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7, # 0.0 - 1.0, lower is more deterministic
    )

    chain = prompt_template | llm_openai

    response = chain.invoke({"name": "Ali"})
    print(response.content)


if __name__ == "__main__":
    main()

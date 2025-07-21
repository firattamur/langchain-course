from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

class TranslationOutput(BaseModel):
    translation: str = Field(..., description="The translated text in Turkish.")
    inp_language: str = Field(..., description="The input language of the text.")
    out_language: str = Field(..., description="The output language of the translation.")


def main() -> None:
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template="Translate the following English text to Turkish: {input}",
    )

    llm_ollama = ChatOllama(model="qwen3:1.7b", temperature=0.7)
    llm_with_structured_output = llm_ollama.with_structured_output(TranslationOutput)

    output_parser_str = StrOutputParser()

    chain_with_str_output = prompt_template | llm_with_structured_output
    response = chain_with_str_output.invoke(
        {"input": "Hello, how are you?"}
    )

    if isinstance(response, TranslationOutput):
        print(f"Translation: {response.translation}")
        print(f"Input Language: {response.inp_language}")
        print(f"Output Language: {response.out_language}")
    else:
        print("Unexpected response format:", response)



if __name__ == "__main__":
    main()

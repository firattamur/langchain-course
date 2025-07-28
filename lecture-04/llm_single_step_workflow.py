from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def main(review_text: str) -> None:
    class ReviewSentiment(BaseModel):
        sentiment: str = Field(
            description="The sentiment of the review, either 'positive', 'negative', or 'neutral'."
        ) 

    prompt_template = PromptTemplate(
        input_variables=["review"],
        template="""
        Analyze the sentiment of this product review:
        Classify it as 'positive', 'negative', or 'neutral'.

        Review Text:
        {review}
        """
    )
    llm_ollama = ChatOllama(model="qwen3:1.7b", temperature=0.7).with_structured_output(ReviewSentiment)

    chain = prompt_template | llm_ollama
    response: ReviewSentiment = chain.invoke({"review": review_text})

    print(f"Single Step LLM Workflow Response: {response.sentiment}")


if __name__ == "__main__":
    review_text = "This product is amazing! It exceeded my expectations in every way. Highly recommend it to anyone looking for quality and performance."
    main(review_text=review_text)

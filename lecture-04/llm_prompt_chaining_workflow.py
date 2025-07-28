from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def main(product: str, target_language: str) -> dict:    
    class MarketingCopy(BaseModel):
        headline: str = Field(description="Catchy headline")
        body: str = Field(description="Marketing copy body")
        call_to_action: str = Field(description="Call to action")
        
    class TranslatedCopy(BaseModel):
        translated_headline: str = Field(description="Translated headline")
        translated_body: str = Field(description="Translated body")
        translated_cta: str = Field(description="Translated call to action")
    
    llm = ChatOllama(model="qwen3:1.7b", temperature=0.8)
    
    copy_prompt = PromptTemplate(
        input_variables=["product"],
        template="""
        Create compelling marketing copy for: {product}
        
        Include:
        - Attention-grabbing headline
        - Persuasive body text (2-3 sentences)
        - Strong call to action
        
        Make it engaging and sales-focused.
        """
    )
    
    copy_chain = copy_prompt | llm.with_structured_output(MarketingCopy)
        
    marketing_copy = copy_chain.invoke({"product": product})
    
    def validate_marketing_copy_gate(copy: MarketingCopy) -> MarketingCopy:        
        issues = []
        
        if len(copy.headline.split()) < 3:
            issues.append("Headline too short")
            
        action_words = ["buy", "get", "try", "order", "download", "sign up", "learn more"]
        if not any(word in copy.call_to_action.lower() for word in action_words):
            issues.append("Call to action not compelling enough")
            
        if len(copy.body.split()) < 10:
            issues.append("Body text too short")
        
        if issues:
            print(f"❌ Gate FAILED: {'; '.join(issues)}")
            raise ValueError(f"Marketing copy validation failed: {issues}")
            
        print("✅ Gate PASSED: Marketing copy approved")
        return copy
    
    validated_copy = validate_marketing_copy_gate(marketing_copy)
    
    translation_prompt = PromptTemplate(
        input_variables=["headline", "body", "cta", "language"],
        template="""
        Translate this marketing copy to {language}:
        
        Headline: {headline}
        Body: {body}
        Call to Action: {cta}
        
        Maintain the marketing tone and persuasive impact.
        Ensure cultural appropriateness for {language} speakers.
        """
    )
    
    translation_chain = translation_prompt | llm.with_structured_output(TranslatedCopy)
    
    translated = translation_chain.invoke({
        "headline": validated_copy.headline,
        "body": validated_copy.body,
        "cta": validated_copy.call_to_action,
        "language": target_language
    })

    print(f"📢 Translated Headline: {translated.translated_headline}")
    print(f"📄 Translated Body: {translated.translated_body}")
    print(f"📣 Translated Call to Action: {translated.translated_cta}")




if __name__ == "__main__":
    product_name = "Smart Home Security Camera"
    target_language = "Spanish"

    main(product=product_name, target_language=target_language)


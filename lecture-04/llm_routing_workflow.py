from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
 

def main(customer_query: str) -> None:    
    class QueryClassification(BaseModel):
        category: str = Field(description="Query category: technical, billing, general, refund")
        confidence: float = Field(description="Classification confidence 0-1")
        complexity: str = Field(description="simple, medium, complex")

    class TechnicalResponse(BaseModel):
        solution: str = Field(description="Technical solution")
        steps: list[str] = Field(description="Step-by-step instructions")

    class BillingResponse(BaseModel):
        explanation: str = Field(description="Billing explanation")
        next_action: str = Field(description="What customer should do next")

    class GeneralResponse(BaseModel):
        answer: str = Field(description="General answer")
        helpful_links: list[str] = Field(description="Helpful resources")
    
    llm = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    
    classifier_prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        Classify this customer service query into categories:
        - technical: password, login, bugs, features, app issues
        - billing: payments, invoices, subscriptions, charges, pricing
        - general: general questions, how-to, information, features
        - refund: returns, refunds, cancellations, money back
        
        Also assess complexity: simple, medium, complex
        
        Customer Query: {query}
        
        Provide classification with confidence score.
        """
    )
    
    classifier = classifier_prompt | llm.with_structured_output(QueryClassification)
    classification = classifier.invoke({"query": customer_query})
    
    def handle_technical(query: str) -> dict:
        tech_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a technical support specialist with expertise in troubleshooting.
            Provide detailed technical help with clear step-by-step solutions.
            Focus on practical troubleshooting steps the user can follow.
            
            Technical Issue: {query}
            
            Provide solution and actionable steps.
            """
        )
        tech_chain = tech_prompt | llm.with_structured_output(TechnicalResponse)
        result = tech_chain.invoke({"query": query})
        return {
            "handler": "Technical Support",
            "solution": result.solution,
            "steps": result.steps,
            "response_type": "technical"
        }
    
    def handle_billing(query: str) -> dict:
        billing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a billing specialist who handles payment and subscription questions.
            Explain billing matters clearly and suggest specific next steps.
            Be empathetic and provide actionable guidance.
            
            Billing Question: {query}
            
            Provide clear explanation and next action for customer.
            """
        )
        billing_chain = billing_prompt | llm.with_structured_output(BillingResponse)
        result = billing_chain.invoke({"query": query})
        return {
            "handler": "Billing Department",
            "explanation": result.explanation,
            "next_action": result.next_action,
            "response_type": "billing"
        }
    
    def handle_general(query: str) -> dict:
        general_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are a friendly customer service representative providing general support.
            Give helpful, informative answers with a positive tone.
            Include useful resources or links when appropriate.
            
            Customer Question: {query}
            
            Provide helpful answer and relevant resources.
            """
        )
        general_chain = general_prompt | llm.with_structured_output(GeneralResponse)
        result = general_chain.invoke({"query": query})
        return {
            "handler": "General Support",
            "answer": result.answer,
            "resources": result.helpful_links,
            "response_type": "general"
        }
    
    def handle_refund(query: str) -> dict:
        return {
            "handler": "Refund Department",
            "message": "Your refund request has been escalated to our specialized refund team. You'll receive a response within 24 hours with next steps.",
            "ticket_created": True,
            "estimated_response": "24 hours",
            "response_type": "refund"
        }
    
    # Step 3: Route to appropriate handler
    routing_table = {
        "technical": handle_technical,
        "billing": handle_billing,
        "general": handle_general,
        "refund": handle_refund
    }
    
    handler = routing_table.get(classification.category, handle_general)
    result = handler(customer_query)

    print(f"🔍 Query Category: {classification.category} (Confidence: {classification.confidence:.2f})")
    print(f"🛠️ Handler: {handler.__name__}")
    print(f"📄 Response: {result}")



if __name__ == "__main__":
    customer_query = "I need help with my billing issue regarding a recent charge."
    main(customer_query=customer_query)

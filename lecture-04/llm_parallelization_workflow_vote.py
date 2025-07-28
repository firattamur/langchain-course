from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableMap


def main(code_snippet: str) -> None:
    class SecurityAssessment(BaseModel):
        has_vulnerability: bool = Field(description="Code has security issue")
        issue_type: str = Field(description="Type of security issue")
        confidence: float = Field(description="Confidence 0-1")
    
    reviewer_prompts = {
        "sql_expert": PromptTemplate(
            input_variables=["code"],
            template="As an SQL injection expert, review this code for vulnerabilities: {code}"
        ),
        "auth_expert": PromptTemplate(
            input_variables=["code"], 
            template="As an authentication expert, review this code for security issues: {code}"
        ),
        "general_expert": PromptTemplate(
            input_variables=["code"],
            template="As a general security expert, review this code for any vulnerabilities: {code}"
        )
    }
    
    llm = ChatOllama(model="qwen3:1.7b", temperature=0.2) 
    
    voting_chains = {
        name: prompt | llm.with_structured_output(SecurityAssessment)
        for name, prompt in reviewer_prompts.items()
    }
    
    parallel_voting = RunnableMap(voting_chains)

    reviews = parallel_voting.invoke({"code": code_snippet})
    
    vulnerability_votes = sum(1 for review in reviews.values() if review.has_vulnerability)
    total_reviewers = len(reviews)
    has_majority_vulnerability = vulnerability_votes > total_reviewers / 2
    
    findings = [
        f"{reviewer}: {'VULN' if review.has_vulnerability else 'SAFE'} - {review.issue_type}"
        for reviewer, review in reviews.items()
    ]

    print(f"Total Reviewers: {total_reviewers}")
    print(f"Vulnerability Votes: {vulnerability_votes}")
    print(f"Majority Vulnerability: {has_majority_vulnerability}")
    print(f"Findings: {findings}")
    print(f"Consensus: {'VULNERABLE' if has_majority_vulnerability else 'SAFE'}")
    


if __name__ == "__main__":
    code_snippet = """
    def login(username, password):
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        return db.execute(query)
    """
    main(code_snippet=code_snippet)


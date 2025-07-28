from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def main(code_task: str, max_iterations: int = 3) -> None:    
    class CodeGeneration(BaseModel):
        code: str = Field(description="Generated code")
        explanation: str = Field(description="How the code works")
        language: str = Field(description="Programming language used")
        complexity: str = Field(description="low, medium, high")

    class CodeReview(BaseModel):
        functionality_score: int = Field(description="Does it work correctly? 1-10")
        quality_score: int = Field(description="Code quality and style 1-10")
        performance_score: int = Field(description="Performance efficiency 1-10")
        issues: list[str] = Field(description="Specific issues found")
        suggestions: list[str] = Field(description="Improvement suggestions")
        is_production_ready: bool = Field(description="Ready for production use")

    class CodeOptimization(BaseModel):
        optimized_code: str = Field(description="Improved code")
        improvements_made: list[str] = Field(description="Specific improvements")
        performance_impact: str = Field(description="Expected performance impact")
        
    generator_llm = ChatOllama(model="qwen3:1.7b", temperature=0.3)
    reviewer_llm  = ChatOllama(model="qwen3:1.7b", temperature=0.1)
    
    code_prompt = PromptTemplate(
        input_variables=["task"],
        template="""
        Write clean, efficient code for this task:
        
        Task: {task}
        
        Provide working code with proper structure and comments.
        Focus on correctness, readability, and efficiency.
        """
    )
    
    review_prompt = PromptTemplate(
        input_variables=["task", "code"],
        template="""
        Review this code thoroughly for a software development project:
        
        Original Task: {task}
        Code to Review: {code}
        
        Evaluate:
        - Functionality: Does it solve the problem correctly?
        - Quality: Is it well-structured, readable, maintainable?
        - Performance: Is it efficient and optimized?
        - Best practices: Does it follow coding standards?
        
        Provide specific issues and actionable suggestions for improvement.
        """
    )
    
    optimize_prompt = PromptTemplate(
        input_variables=["task", "code", "review"],
        template="""
        Optimize this code based on the review feedback:
        
        Original Task: {task}
        Current Code: {code}
        Review Feedback: {review}
        
        Improve the code addressing the reviewer's suggestions.
        Focus on fixing issues while maintaining functionality.
        """
    )
    
    iterations = []
    
    generator = code_prompt | generator_llm.with_structured_output(CodeGeneration)
    initial_code = generator.invoke({"task": code_task})
    current_code = initial_code.code
    
    iterations.append({
        "iteration": 0,
        "type": "generation",
        "code": initial_code.code,
        "explanation": initial_code.explanation,
        "language": initial_code.language
    })
    
    reviewer = review_prompt | reviewer_llm.with_structured_output(CodeReview)
    optimizer = optimize_prompt | generator_llm.with_structured_output(CodeOptimization)
    
    for iteration in range(max_iterations):
        review = reviewer.invoke({"task": code_task, "code": current_code})
        
        iteration_data = {
            "iteration": iteration + 1,
            "type": "review",
            "functionality_score": review.functionality_score,
            "quality_score": review.quality_score,
            "performance_score": review.performance_score,
            "issues": review.issues,
            "suggestions": review.suggestions,
            "production_ready": review.is_production_ready
        }
        
        if review.is_production_ready:
            iterations.append(iteration_data)
            break
        
        review_text = " | ".join(review.suggestions)
        optimization = optimizer.invoke({
            "task": code_task,
            "code": current_code,
            "review": review_text
        })
        
        current_code = optimization.optimized_code
        
        iteration_data["optimization"] = {
            "code": optimization.optimized_code,
            "improvements": optimization.improvements_made,
            "performance_impact": optimization.performance_impact
        }
        
        iterations.append(iteration_data)
    
    print("🔄 EVALUATOR & OPTIMIZER WORKFLOW RESULTS")

    print("=" * 60)
    print(f"\nTask: {code_task}")
    print(f"Initial Code:\n{initial_code.code}\n")
    print(f"Explanation: {initial_code.explanation}")
    print(f"Language: {initial_code.language}")
    print(f"Complexity: {initial_code.complexity}\n")

    for iteration in iterations:
        print(f"Iteration {iteration['iteration']} - {iteration['type'].upper()}")
        
        if iteration['type'] == "generation":
            print(f"Generated Code:\n{iteration['code']}\n")
            print(f"Explanation: {iteration['explanation']}")
            print(f"Language: {iteration['language']}\n")
        
        elif iteration['type'] == "review":
            print(f"Functionality Score: {iteration['functionality_score']}/10")
            print(f"Quality Score: {iteration['quality_score']}/10")
            print(f"Performance Score: {iteration['performance_score']}/10")
            print("Issues Found:")
            for issue in iteration['issues']:
                print(f"  - {issue}")
            print("Suggestions:")
            for suggestion in iteration['suggestions']:
                print(f"  - {suggestion}")
            print(f"Production Ready: {'Yes' if iteration['production_ready'] else 'No'}\n")
            
            if 'optimization' in iteration:
                opt = iteration['optimization']
                print("Optimized Code:")
                print(opt['code'])
                print("Improvements Made:")
                for improvement in opt['improvements']:
                    print(f"  - {improvement}")
                print(f"Performance Impact: {opt['performance_impact']}\n")

    print("=" * 60)


if __name__ == "__main__":
    code_task = "Write a Python function to calculate Fibonacci numbers up to n"    
    main(code_task=code_task, max_iterations=3)

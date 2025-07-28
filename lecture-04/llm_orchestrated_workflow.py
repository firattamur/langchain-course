from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


def main(complex_task: str) -> None:    
    class TaskBreakdown(BaseModel):
        subtasks: list[dict[str, str]] = Field(description="List of subtasks with type and description")
        dependencies: list[str] = Field(description="Dependencies between subtasks")
        approach: str = Field(description="Overall approach for coordination")
    
    class WorkerResult(BaseModel):
        subtask_id: str = Field(description="ID of completed subtask")
        result: str = Field(description="Worker's result for this subtask") 
        status: str = Field(description="completed, failed, needs_revision")
        confidence: float = Field(description="Confidence in result 0-1")
    
    class FinalSynthesis(BaseModel):
        integrated_result: str = Field(description="Final integrated result")
        quality_assessment: str = Field(description="Quality assessment of final result")
        completeness: str = Field(description="Assessment of task completeness")
        recommendations: list[str] = Field(description="Additional recommendations")
    
    orchestrator_llm = ChatOllama(model="qwen3:1.7b", temperature=0.2) 
    worker_llm = ChatOllama(model="qwen3:1.7b", temperature=0.4) 
    
    print("🎯 ORCHESTRATOR-WORKERS WORKFLOW")
    print("=" * 50)
    print(f"📋 Complex Task: {complex_task}")
    
    print("\n🧠 ORCHESTRATOR: Analyzing and breaking down task...")
    
    orchestrator_prompt = PromptTemplate(
        input_variables=["task"],
        template="""
        You are an orchestrator LLM. Analyze this complex task and break it down into specific subtasks.
        Consider what types of specialized work are needed and how they relate to each other.
        
        Complex Task: {task}
        
        Break this down into:
        - Specific subtasks that different workers should handle
        - What type of specialist each subtask needs (researcher, analyst, writer, coder, etc.)
        - Dependencies between subtasks
        - Overall coordination approach
        
        Be flexible - the number and nature of subtasks should match what THIS specific task requires.
        """
    )
    
    breakdown_chain = orchestrator_prompt | orchestrator_llm.with_structured_output(TaskBreakdown)
    breakdown = breakdown_chain.invoke({"task": complex_task})
    
    print(f"✅ Task broken into {len(breakdown.subtasks)} dynamic subtasks:")
    for i, subtask in enumerate(breakdown.subtasks, 1):
        print(f"   {i}. [{subtask['type']}] {subtask['description']}")
    
    print(f"🔗 Dependencies: {', '.join(breakdown.dependencies) if breakdown.dependencies else 'None'}")
    print(f"📋 Approach: {breakdown.approach}")
    
    # STEP 2: WORKERS EXECUTE THEIR ASSIGNED SUBTASKS
    print("\n👥 WORKERS: Executing assigned subtasks...")
    
    worker_results = []
    
    for i, subtask in enumerate(breakdown.subtasks):
        print(f"\n🔨 Worker {i+1} ({subtask['type']}) starting...")
        
        worker_prompt = PromptTemplate(
            input_variables=["original_task", "subtask_type", "subtask_description", "context"],
            template="""
            You are a specialized {subtask_type} worker. Complete this specific subtask as part of a larger project.
            
            Original Complex Task: {original_task}
            Your Specialist Role: {subtask_type}
            Your Specific Subtask: {subtask_description}
            Context: {context}
            
            Execute this subtask with expertise in your specialized area.
            Provide thorough, professional results that will integrate with other workers' outputs.
            """
        )
        
        worker_chain = worker_prompt | worker_llm.with_structured_output(WorkerResult)
        
        # Execute worker task
        result = worker_chain.invoke({
            "original_task": complex_task,
            "subtask_type": subtask['type'],
            "subtask_description": subtask['description'],
            "context": breakdown.approach
        })
        
        result.subtask_id = f"task_{i+1}_{subtask['type']}"
        worker_results.append(result)
        
        print(f"   ✅ Completed with {result.confidence:.1%} confidence ({result.status})")
    
    print("\n🔄 ORCHESTRATOR: Synthesizing worker results...")
    
    synthesis_prompt = PromptTemplate(
        input_variables=["original_task", "breakdown", "worker_results"],
        template="""
        You are the orchestrator. Integrate all worker results into a final comprehensive solution.
        
        Original Task: {original_task}
        Task Breakdown: {breakdown}
        Worker Results: {worker_results}
        
        Synthesize all worker outputs into:
        - A cohesive, integrated final result
        - Quality assessment of the overall solution
        - Completeness evaluation
        - Any additional recommendations
        
        Ensure the final result addresses the original complex task completely.
        """
    )
    
    synthesis_chain = synthesis_prompt | orchestrator_llm.with_structured_output(FinalSynthesis)
    
    results_summary = []
    for result in worker_results:
        results_summary.append(f"[{result.subtask_id}]: {result.result}")
    
    final_result = synthesis_chain.invoke({
        "original_task": complex_task,
        "breakdown": breakdown.approach,
        "worker_results": " | ".join(results_summary)
    })
    
    print("✅ Integration complete!")
    print(f"📊 Quality: {final_result.quality_assessment}")
    print(f"📋 Completeness: {final_result.completeness}")
    
    print("\n🎯 FINAL INTEGRATED RESULT:")
    print(f"{final_result.integrated_result}")
    
    if final_result.recommendations:
        print("\n💡 ADDITIONAL RECOMMENDATIONS:")
        for i, rec in enumerate(final_result.recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("=" * 50)


if __name__ == "__main__":
    complex_task = "Research the impact of climate change on global agriculture and provide a comprehensive report."
    main(complex_task=complex_task)

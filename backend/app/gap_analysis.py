from typing import List, Dict
from .llm_wrapper import get_llm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def identify_research_gaps(
    answer: str,
    related_papers: List[Dict],
    question: str
) -> List[Dict]:
    """
    Identify research gaps based on answer and related literature.

    Args:
        answer: Generated answer
        related_papers: List of related paper dictionaries
        question: Original question

    Returns:
        List of research gap dictionaries
    """
    # Format related papers
    papers_text = "\n".join([
        f"- {p['title']} ({p.get('year', 'N/A')}): {p.get('authors', 'N/A')}"
        for p in related_papers[:5]  # Top 5
    ])

    # Create optimized prompt for gap analysis - shorter and more structured
    prompt_template = PromptTemplate(
        input_variables=["answer", "related_papers", "question"],
        template="""You are a research analyst. Identify 3-5 specific research gaps.

QUESTION: {question}
ANSWER: {answer}
LITERATURE: {related_papers}

TASK: List gaps with descriptions and approaches.

FORMAT (exactly):
Gap 1:
Description: [1-2 sentences]
Approach: [1-2 sentences]

Gap 2:
Description: [1-2 sentences]
Approach: [1-2 sentences]

[Continue for 3-5 gaps]

Be specific and actionable. Focus on researchable gaps.
"""
    )

    # Generate gaps
    try:
        llm = get_llm()
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Try different invocation methods for compatibility
        try:
            gaps_text = chain.run(
                answer=answer,
                related_papers=papers_text,
                question=question
            )
        except Exception:
            # Try alternative method
            result = chain.invoke({
                "answer": answer,
                "related_papers": papers_text,
                "question": question
            })
            if isinstance(result, dict):
                gaps_text = result.get("text", str(result))
            else:
                gaps_text = str(result)
    except Exception as e:
        print(f"Error in gap analysis: {e}")
        # Return empty gaps on error
        return []

    # Ensure gaps_text is a string
    if not gaps_text or not isinstance(gaps_text, str):
        return []

    # Parse gaps (simple parsing - could be improved)
    gaps = []
    lines = gaps_text.split("\n")
    current_gap = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if it's a numbered item
        if line and line[0].isdigit():
            if current_gap:
                gaps.append(current_gap)
            current_gap = {"description": line, "suggestions": []}
        elif current_gap and ("approach" in line.lower() or "experiment" in line.lower()):
            current_gap["suggestions"].append(line)
        elif current_gap:
            current_gap["description"] += " " + line

    if current_gap:
        gaps.append(current_gap)

    return gaps[:5]  # Return top 5


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

    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["answer", "related_papers", "question"],
        template="""You are a research analyst. Based on the following answer to a research question and the related literature, identify 3-5 research gaps and suggest potential experimental approaches.

Question: {question}

Answer: {answer}

Related Literature:
{related_papers}

Identify research gaps (areas not fully addressed) and suggest:
1. What questions remain unanswered?
2. What methodological approaches could address these gaps?
3. What experiments or studies would be valuable?

Format your response as a numbered list of gaps, each with:
- Gap description
- Suggested approach/experiment

Be specific and actionable.
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


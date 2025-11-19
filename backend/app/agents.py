from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Dict
from sqlalchemy.orm import Session

from .llm_wrapper import get_llm
from .rag import format_context_for_llm, retrieve_chunks
from .related_literature import find_related_papers
from .gap_analysis import identify_research_gaps

# Initialize LLM
_llm = None


def get_llm_instance():
    """Get or create LLM instance"""
    global _llm
    if _llm is None:
        _llm = get_llm()
        if _llm is None:
            raise RuntimeError(
                "LLM instance is None. Please check your LLM_PROVIDER configuration and ensure "
                "OPENAI_API_KEY is set if using OpenAI, or configure Ollama/local LLM properly."
            )
    return _llm


def answer_with_rag(
    chunks: List[Dict],
    question: str
) -> str:
    """
    Generate answer using RAG with retrieved chunks.

    Args:
        chunks: List of retrieved chunk dictionaries
        question: User question

    Returns:
        Generated answer with citations
    """
    if not chunks:
        return "No relevant content found in the uploaded documents to answer this question."

    # Format context with reasonable limits to balance quality and speed
    # Use top 8 chunks and allow up to 800 chars per chunk for comprehensive context
    limited_chunks = chunks[:8]  # Use top 8 chunks for better coverage
    context = format_context_for_llm(chunks, max_chunk_length=800)

    # Create optimized prompt for better responses
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert academic research assistant. Your task is to provide clear, accurate, and well-cited answers based on the provided research context.

CONTEXT FROM RESEARCH DOCUMENTS:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the context carefully and identify the most relevant information
2. Provide a comprehensive but concise answer (aim for 150-300 words)
3. Include citations in the format [Document Title, pX-Y] for every claim or fact you reference
4. If multiple sources support the same point, cite all relevant sources
5. If the context doesn't contain enough information, clearly state what is missing
6. Structure your answer with clear paragraphs
7. Be specific and avoid vague statements

ANSWER:
"""
    )

    # Create chain
    try:
        llm = get_llm_instance()
        print(f"Using LLM type: {llm._llm_type}")
        
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Generate answer - try different methods for compatibility
        try:
            result = chain.run(context=context, question=question)
        except Exception as chain_error:
            # Try alternative invocation method
            print(f"Chain.run() failed: {chain_error}, trying invoke()")
            try:
                result = chain.invoke({"context": context, "question": question})
                if isinstance(result, dict):
                    result = result.get("text", str(result))
            except Exception as invoke_error:
                # Last resort: call LLM directly using invoke (newer LangChain API)
                print(f"Chain.invoke() failed: {invoke_error}, calling LLM directly")
                full_prompt = prompt_template.format(context=context, question=question)
                try:
                    # Try invoke() method (newer LangChain)
                    result = llm.invoke(full_prompt)
                    if isinstance(result, dict):
                        result = result.get("content", result.get("text", str(result)))
                except AttributeError:
                    # Fallback to generate() if invoke doesn't exist
                    try:
                        result = llm.generate([full_prompt])
                        if hasattr(result, 'generations') and result.generations:
                            result = result.generations[0][0].text
                    except Exception as gen_error:
                        raise RuntimeError(f"Failed to call LLM: {gen_error}")
        
        if not result:
            return "Unable to generate an answer. Please check your LLM configuration."
        
        # Handle different return types
        if isinstance(result, dict):
            result = result.get("text", str(result))
        
        return str(result).strip()
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error generating answer with LLM: {error_trace}")
        raise RuntimeError(f"Error generating answer with LLM: {str(e)}")


def full_rag_pipeline(
    question: str,
    db: Session,
    k: int = 8
) -> Dict:
    """
    Complete RAG pipeline: retrieve, answer, find related papers, identify gaps.

    Returns:
        Dict with answer, sources, related_papers, gaps
    """
    # Step 1: Retrieve relevant chunks
    try:
        chunks = retrieve_chunks(question, db, k=k)
    except ValueError as e:
        # Re-raise with more context
        raise ValueError(str(e))
    except Exception as e:
        raise RuntimeError(f"Error retrieving chunks: {str(e)}")

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded documents.",
            "sources": [],
            "related_papers": [],
            "gaps": []
        }

    # Step 2: Generate answer
    try:
        answer = answer_with_rag(chunks, question)
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {str(e)}")

    # Step 3: Find related papers (with error handling)
    try:
        related_papers = find_related_papers(question, answer, db, top_k=5)
    except Exception as e:
        # Don't fail the whole request if related papers fail
        print(f"Warning: Error finding related papers: {str(e)}")
        related_papers = []

    # Step 4: Identify research gaps (with error handling)
    try:
        gaps = identify_research_gaps(answer, related_papers, question)
    except Exception as e:
        # Don't fail the whole request if gap analysis fails
        print(f"Warning: Error identifying research gaps: {str(e)}")
        gaps = []

    # Format sources
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "page_range": f"{c['page_start']}-{c['page_end']}",
            "snippet": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
            "similarity_score": c["similarity_score"]
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "related_papers": related_papers,
        "gaps": gaps
    }


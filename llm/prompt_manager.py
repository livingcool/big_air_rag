# llm/prompt_manager.py (GEMINI EDITION)

import os
import re
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Configuration ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# --- Client Initialization ---
GEMINI_AVAILABLE = False
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        # Verify init by listing models (optional check, but good for debugging)
        # genai.list_models() 
        GEMINI_AVAILABLE = True
        print(f"Prompt Manager: Google Gemini initialized. Using {GEMINI_MODEL_NAME}.")
    else:
        print("Prompt Manager: GEMINI_API_KEY not found. LLM disabled.")
except Exception as e:
    print(f"Prompt Manager Warning: Gemini init failed ({e}).")


# --- Strict System Instruction ---
SYSTEM_INSTRUCTION = """
You are a highly factual document analysis assistant. 
Your task is to answer the user's question STRICTLY based on the provided CONTEXT text.

## CRITICAL RULES
1. **Faithfulness:** Do NOT use outside knowledge. If the answer is not in the context, state "I don't know based on the provided documents."
2. **Citation Mandate:** You MUST cite your sources. Every claim must be followed by a tag in this exact format: [DOC:ID | PAGE:NUMBER | CHUNK:ID].
3. **Format:** Keep the answer concise and professional.
"""

def generate_fallback_answer(query: str, formatted_context: str) -> Dict[str, Any]:
    """
    Fallback if Gemini API fails or key is missing.
    """
    print("LLM: Using Heuristic Fallback (Offline Mode).")
    context_lines = formatted_context.split('\n\n')
    top_evidence = context_lines[:3] if len(context_lines) > 0 else ["No context available."]
    
    answer_text = (
        "**[System Notice: LLM Generation Failed. Showing Raw Evidence]**\n\n"
        "Based on the highest-ranked documents, here is the relevant information:\n\n"
    )
    for snippet in top_evidence:
        answer_text += f"> {snippet}\n\n"

    citation_pattern = r"\[DOC:[^\]]+\]"
    citations = sorted(list(set(re.findall(citation_pattern, formatted_context))))

    return {
        "answer": answer_text,
        "citations": citations,
        "model_used": "Heuristic-Fallback"
    }


def generate_answer_with_citations(query: str, formatted_context: str) -> Dict[str, Any]:
    """
    Generates answer using Google Gemini 1.5 Flash.
    """
    if not GEMINI_AVAILABLE:
        return generate_fallback_answer(query, formatted_context)

    # Construct the full prompt
    full_prompt = f"""
    {SYSTEM_INSTRUCTION}

    CONTEXT:
    ---
    {formatted_context}
    ---

    USER QUESTION: {query}
    """
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        
        # Generate content
        print(f"LLM: Calling Gemini with {len(formatted_context)} chars of context...")
        response = model.generate_content(full_prompt)
        
        # Extract text
        answer_text = response.text
        
        # Post-process citations
        citation_pattern = r"\[DOC:[^\]]+\]"
        citations = sorted(list(set(re.findall(citation_pattern, answer_text))))

        return {
            "answer": answer_text,
            "citations": citations,
            "model_used": GEMINI_MODEL_NAME
        }
        
    except Exception as e:
        print(f"ERROR during Gemini API call: {e}")
        return generate_fallback_answer(query, formatted_context)
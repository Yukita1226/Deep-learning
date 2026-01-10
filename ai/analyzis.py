import os
import sys
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()


def run_grading(input_text: str):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    if not GROQ_API_KEY or not TAVILY_API_KEY:
        return "CRITICAL ERROR: Keys missing."

    client = Groq(api_key=GROQ_API_KEY)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    try:
 
        router_prompt = f"Does this need real-time web search for factual grading? Answer 'YES' or 'NO' only. Input: {input_text}"
        router_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0
        )
        
        needs_search = "YES" in router_res.choices[0].message.content.upper()
        web_info = ""

  
        if needs_search:
            search_query_prompt = (
                "Extract 3-5 essential Thai keywords for a search engine from this text. "
                "Output ONLY the keywords separated by spaces. No numbers, no headers, no intro."
                f"\nText: {input_text}"
            )
            search_query_res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": search_query_prompt}],
                temperature=0 
            )
            optimized_query = search_query_res.choices[0].message.content.strip()[:350]
            
            search_res = tavily.search(query=optimized_query, search_depth="basic", max_results=2)
            web_info = "\n".join([r['content'] for r in search_res['results']])

      
        system_content = f"""
You are a UNIVERSITY-LEVEL GRADING EXPERT. 
Rules:
1. Identify type (CALCULATION, ESSAY, etc.)
2. Use the provided WEB CONTEXT as the absolute source of truth.
3. If the answer contradicts WEB CONTEXT, score must be below 50%.
4. Respond in the SAME LANGUAGE as the user.

WEB CONTEXT:
{web_info if web_info else "No external data needed."}

OUTPUT FORMAT:
Score: <0-100>%
Reason: <concise evaluation>
"""

        final_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": input_text}
            ],
            temperature=0.1 
        )

       
        return final_res.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    test_text = "Question: 1+1? Answer: 3"
    print(run_grading(test_text))
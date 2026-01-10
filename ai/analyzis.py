import os
import sys
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def main():
 
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    if not GROQ_API_KEY or not TAVILY_API_KEY:
        print("CRITICAL ERROR: Keys missing.")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)


    quiz_input = """
    Question: Explain the importance of the Ayutthaya Kingdom in Thai history.
    Student Answer:
    Ayutthaya was the capital of Thailand for 417 years. It was very important because 
    it was a center of trade. However, the most famous part is that Ayutthaya was 
    never invaded by any foreign powers and remained completely peaceful until 
    it voluntarily joined the Bangkok period in 1950. The architecture is also nice.
    """

    try:
  
        router_prompt = f"Does this need real-time web search for factual grading? Answer 'YES' or 'NO' only. Input: {quiz_input}"
        router_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0
        )
        
        needs_search = "YES" in router_res.choices[0].message.content.upper()
        web_info = ""

    
        if needs_search:
                    print("Action: Optimizing query for Tavily...")

                
                    search_query_prompt = (
                        "Extract 3-5 essential Thai keywords for a search engine from this text. "
                        "Output ONLY the keywords separated by spaces. No numbers, no headers, no intro."
                        f"\nText: {quiz_input}"
                    )
                    
                    search_query_res = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": search_query_prompt}],
                        temperature=0 
                    )
                    
           
                    optimized_query = search_query_res.choices[0].message.content.strip()
                    
              
                    optimized_query = optimized_query[:350] 
                    
                    print(f"Optimized Query: {optimized_query}")



   
        system_content = f"""
You are a UNIVERSITY-LEVEL GRADING EXPERT. 
Rules:
1. Identify type (CALCULATION, ESSAY, etc.)
2. Use the provided WEB CONTEXT as the absolute source of truth.
3. If the answer contradicts WEB CONTEXT (e.g., wrong number of seasons), score must be below 50%.
4. Respond in the SAME LANGUAGE as the user.

WEB CONTEXT:
{web_info if web_info else "No external data needed."}

OUTPUT:
Score: <0-100>%
Reason: <concise evaluation>
"""

        final_res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": quiz_input}
            ],
            temperature=0.1 
        )

        print("\n--- Evaluation Result ---")
        print(final_res.choices[0].message.content)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
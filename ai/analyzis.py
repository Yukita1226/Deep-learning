import os
import sys
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def main():
    # โหลด Keys
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
    
    if not GROQ_API_KEY or not TAVILY_API_KEY:
        print("CRITICAL ERROR: Keys missing.")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    # --- SINGLE INPUT STRING ---
    # คุณสามารถเปลี่ยนข้อความในนี้ได้เลย ระบบจะปรับตัวตามเนื้อหาเอง
    quiz_input = "คำถาม: ประเทศไทยมีกี่ฤดู และแต่ละฤดูเป็นอย่างไร | คำตอบจากผู้เรียน: ประเทศไทยมี 4 ฤดู มีฤดูใบไม้ผลิด้วย"

    try:
        # 1. Router: ตรวจสอบว่าต้องใช้ข้อมูลภายนอกหรือไม่
        router_prompt = f"Does this need real-time web search for factual grading? Answer 'YES' or 'NO' only. Input: {quiz_input}"
        router_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0
        )
        
        needs_search = "YES" in router_res.choices[0].message.content.upper()
        web_info = ""

    
        if needs_search:
            print("Action: Searching Tavily for facts...")
            # ดึงข้อมูลมาเป็น Context เพื่อลด Hallucination
            search_res = tavily.search(query=quiz_input, search_depth="basic", max_results=2)
            web_info = "\n".join([r['content'] for r in search_res['results']])

   
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
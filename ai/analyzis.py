import os
import sys
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 

def main():
    quizandanz = (
        "ประเทศไทยมีกี่ฤดู และแต่ละฤดูมีลักษณะอย่างไร"
        "ประเทศไทยมีทั้งหมด 4 ฤดู ได้แก่ ฤดูร้อน ฤดูฝน ฤดูหนาว และฤดูใบไม้ผลิฤดูร้อนจะมีอากาศร้อนจัดและแดดแรงฤดูฝนจะมีฝนตกชุกและอากาศชื้นฤดูหนาวอากาศจะเย็นลงเล็กน้อยส่วนฤดูใบไม้ผลิเป็นช่วงที่อากาศกำลังสบาย ไม่ร้อนหรือหนาวเกินไป"
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: GROQ_API_KEY not found.")
        sys.exit(1)

    system_content = f"""
You are a STRICT and ADAPTIVE grading assistant.

Your first step is to IDENTIFY the type of the user's input.
Classify it into ONE primary category:
- CALCULATION (math, logic, strategy, step-based problems)
- ESSAY (writing, explanation, reflection, descriptive answers)
- DEBATE / ARGUMENT (claims, opinions, reasoning, persuasion)
- FACTUAL QA (short factual questions/answers)
- OTHER (specify implicitly in reasoning)

LANGUAGE RULE (MANDATORY):
- You MUST respond in the SAME LANGUAGE as the user's input.
- Do NOT mix languages.

SCORING PRIORITY BY TYPE:

1) CALCULATION:
   - Strategy, method, and final result are CRITICAL.
   - Correct result with flawed reasoning → penalize.
   - Wrong result → score MUST NOT exceed 50%.
   - Major logical error → heavy penalty regardless of clarity.

2) ESSAY:
   - Grammar, clarity, structure, and coherence are PRIMARY.
   - Factual accuracy is important but secondary unless the essay is fact-based.
   - Poor grammar or unclear structure → significant penalty.

3) DEBATE / ARGUMENT:
   - Factual accuracy and strength of reasoning are PRIMARY.
   - Unsupported claims, logical fallacies, or false facts → heavy penalty.
   - Writing style alone MUST NOT raise the score.

4) FACTUAL QA:
   - Factual correctness is ABSOLUTE.
   - Any major factual error → score MUST NOT exceed 60%.

GENERAL RULES (MANDATORY):
- Fluent language MUST NOT compensate for incorrect facts or logic.
- Sounding reasonable but being wrong MUST be penalized.
- Do NOT inflate scores.

OUTPUT FORMAT (EXACT, NO EXTRA TEXT):
Score: <0-100>%
Reason: <2-6 sentences explaining the evaluation based on the identified type>

Be strict, fair, and consistent.

"""

    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": quizandanz}
            ],
        )

        print("Success:")
        print(chat_completion.choices[0].message.content)

    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    main()

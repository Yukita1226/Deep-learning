import os
import sys
from groq import Groq
from dotenv import load_dotenv

# 1. Load the variables
load_dotenv() 

def main():
    # 2. Get the key
    api_key = os.environ.get("GROQ_API_KEY")

    # 3. Graduation Pro-Tip: Explicit Validation
    # This tells you EXACTLY what is wrong instead of a generic Groq error
    if not api_key:
        print("CRITICAL ERROR: GROQ_API_KEY not found.")
        print(f"Current Working Directory: {os.getcwd()}")
        sys.exit(1)

    # 4. Initialize client
    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": "Confirm: respond 'Llama 3 Ready'"}],
            model="llama-3.3-70b-versatile",
        )
        print(f"Success: {chat_completion.choices[0].message.content}")
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    main()
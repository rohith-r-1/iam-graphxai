import re

txt = open('src/llm_reasoning.py', encoding='utf-8').read()

new_func = '''def check_ollama_available():
    if len(GROQ_API_KEY) < 20:
        print("  ERROR: Set your Groq API key.")
        return False
    print(f"  Groq API key        : {GROQ_API_KEY[:8]}...")
    print(f"  Groq model          : {GROQ_MODEL}")
    return True
'''

txt = re.sub(r'def check_ollama_available\(\):.*?return False\n',
             new_func, txt, flags=re.DOTALL)

open('src/llm_reasoning.py', 'w', encoding='utf-8').write(txt)
print('Done')

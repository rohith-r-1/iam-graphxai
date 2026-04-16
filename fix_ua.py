txt = open('src/llm_reasoning.py', encoding='utf-8').read()
txt = txt.replace(
    '"Content-Type":  "application/json",\n            "Authorization": f"Bearer {GROQ_API_KEY}"',
    '"Content-Type":  "application/json",\n            "Authorization": f"Bearer {GROQ_API_KEY}",\n            "User-Agent":    "python-urllib/1.0"'
)
open('src/llm_reasoning.py', 'w', encoding='utf-8').write(txt)
print('Done')

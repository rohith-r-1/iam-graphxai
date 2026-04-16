txt = open('src/api_final.py', encoding='utf-8').read()

# Just count all feature key patterns in the extract function
import re
# Find the function body
start = txt.find('def extract_features_from_policy')
end   = txt.find('\ndef ', start + 10)
body  = txt[start:end]

keys = re.findall(r"'([a-z_]+)':\s+", body)
print(f"Features found: {len(keys)}")
for i, k in enumerate(keys):
    print(f"  {i+1:2d}. {k}")

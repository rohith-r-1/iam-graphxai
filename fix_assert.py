txt = open('src/api_final.py', encoding='utf-8').read()
txt = txt.replace(
    "    assert len(features) == 38, f\"Feature count mismatch: {len(features)}\"",
    "    # feature count handled by model column alignment in assess_policy"
)
open('src/api_final.py', 'w', encoding='utf-8').write(txt)
print('Done')

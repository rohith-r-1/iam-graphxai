txt = open('src/api_final.py', encoding='utf-8').read()
txt = txt.replace(
    "svc_wc       = [a for a in all_actions if a.endswith(':*')]",
    "svc_wc       = [a for a in all_actions if a.endswith(':*') or a == '*']"
)
txt = txt.replace(
    "dang_match   = [a for a in all_actions if any(d in a for d in DANGEROUS)]",
    "dang_match   = [a for a in all_actions if any(d in a for d in DANGEROUS) or a == '*']"
)
txt = txt.replace(
    "esc_match    = [a for a in all_actions if a in ESC]",
    "esc_match    = [a for a in all_actions if a in ESC or a == '*']"
)
open('src/api_final.py', 'w', encoding='utf-8').write(txt)
print('Done')

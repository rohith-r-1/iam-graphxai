txt = open('src/api_final.py', encoding='utf-8').read()
txt = txt.replace(
    "'effective_permission_score':      min((len(admin_match)*3 + len(dang_match) +\n                                               len(wc_actions)*2) / max(n*3, 1), 1.0),",
    "'effective_permission_score':      min((len(admin_match)*3 + len(dang_match) +\n                                               len(wc_actions)*2) / max(n*3, 1), 1.0),\n        'deny_statement_ratio':             len(deny_stmts) / max(len(statements), 1),"
)
open('src/api_final.py', 'w', encoding='utf-8').write(txt)
print('Done')

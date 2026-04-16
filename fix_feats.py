import re

txt = open('src/api_final.py', encoding='utf-8').read()

new_func = '''def extract_features_from_policy(policy: dict) -> dict:
    """Map raw IAM policy JSON to the exact 38 features the RF model was trained on."""
    import math

    statements = []
    if isinstance(policy, dict):
        stmts = policy.get('Statement', policy.get('statement', []))
        if isinstance(stmts, list):   statements = stmts
        elif isinstance(stmts, dict): statements = [stmts]

    all_actions, all_resources, all_conditions = [], [], []
    allow_stmts, deny_stmts = [], []

    for stmt in statements:
        effect    = stmt.get('Effect', stmt.get('effect', 'Allow'))
        actions   = stmt.get('Action',   stmt.get('action',   []))
        resources = stmt.get('Resource', stmt.get('resource', []))
        cond      = stmt.get('Condition',stmt.get('condition',{}))
        if isinstance(actions,   str): actions   = [actions]
        if isinstance(resources, str): resources = [resources]
        all_actions.extend([a.lower() for a in actions])
        all_resources.extend(resources)
        if cond: all_conditions.append(cond)
        (allow_stmts if effect == \'Allow\' else deny_stmts).append(stmt)

    n = max(len(all_actions), 1)

    services = set()
    for a in all_actions:
        p = a.split(\':\')
        if len(p) == 2: services.add(p[0])

    DANGEROUS = [\'iam:passrole\',\'iam:createuser\',\'iam:attachuserpolicy\',\'iam:attachrolepolicy\',
                 \'iam:putuserupolicy\',\'iam:putrolepolicy\',\'iam:createpolicyversion\',
                 \'iam:setdefaultpolicyversion\',\'iam:addusertogroup\',\'iam:createaccesskey\',
                 \'secretsmanager:getsecretvalue\',\'kms:decrypt\',\'ssm:getparameter\',
                 \'sts:assumerole\',\'lambda:invokefunction\']
    ESC = [\'iam:passrole\',\'iam:createpolicyversion\',\'iam:setdefaultpolicyversion\',
           \'iam:createuser\',\'iam:attachuserpolicy\',\'iam:attachrolepolicy\',
           \'iam:putuserupolicy\',\'iam:putrolepolicy\',\'iam:addusertogroup\',\'iam:createaccesskey\']
    IAM_WRITE = [\'iam:createuser\',\'iam:deleteuser\',\'iam:attachuserpolicy\',\'iam:detachuserpolicy\',
                 \'iam:putuserupolicy\',\'iam:deleteuserupolicy\',\'iam:creategroup\',\'iam:addusertogroup\',
                 \'iam:createrolepolicy\',\'iam:putrolepolicy\',\'iam:attachrolepolicy\',
                 \'iam:createrole\',\'iam:deleterole\',\'iam:passrole\',\'iam:createpolicy\',
                 \'iam:createpolicyversion\',\'iam:setdefaultpolicyversion\',\'iam:createaccesskey\']

    wc_actions   = [a for a in all_actions if \'*\' in a]
    dang_match   = [a for a in all_actions if any(d in a for d in DANGEROUS)]
    esc_match    = [a for a in all_actions if a in ESC]
    svc_wc       = [a for a in all_actions if a.endswith(\':*\')]
    iam_write    = [a for a in all_actions if a in IAM_WRITE]
    wc_res       = [r for r in all_resources if r == \'*\']
    arn_res      = [r for r in all_resources if r.startswith(\'arn:\')]

    has_passrole  = int(any(\'iam:passrole\'            in a for a in all_actions))
    has_cpv       = int(any(\'iam:createpolicyversion\'  in a for a in all_actions))
    has_aup       = int(any(\'iam:attachuserpolicy\'     in a for a in all_actions))
    has_mfa       = int(any(\'aws:multifactorauthpresent\' in str(c).lower() for c in all_conditions))
    has_ip        = int(any(\'aws:sourceip\'             in str(c).lower() for c in all_conditions))
    has_time      = int(any(\'aws:currenttime\'          in str(c).lower() for c in all_conditions))
    is_bounded    = int(bool(arn_res) and not bool(wc_res))

    hop = 0 if (esc_match or any(a in [\'*\',\'iam:*\'] for a in all_actions)) else (1 if dang_match else 3)

    # wildcard entropy
    wc_ratio = len(wc_actions) / n
    wc_ent   = -wc_ratio * math.log2(wc_ratio + 1e-9) if wc_ratio > 0 else 0.0

    # specificity score (0-1, higher = more specific)
    spec = len(arn_res) / max(len(all_resources), 1)

    # action diversity (unique / total)
    act_div = len(set(all_actions)) / n

    # resource ARN specificity
    arn_spec = len(arn_res) / max(len(all_resources), 1)

    # permission overlap (wildcards vs total)
    perm_overlap = len(wc_actions) / n

    # cross-service chains (services that can call each other)
    sensitive = [s for s in services if s in [\'iam\',\'sts\',\'kms\',\'secretsmanager\',\'ssm\',\'s3\',\'lambda\']]
    cross_chains = max(len(sensitive) - 1, 0)

    # escalation path approximation
    esc_path_count = len(esc_match)
    min_esc_len    = 1 if esc_match else 99
    esc_techniques = len(set(esc_match))

    # privilege escalation risk score
    esc_score = min((has_passrole * 3 + has_cpv * 2 + has_aup + len(esc_match)) / 7.0, 1.0)

    # condition protection score
    cond_score = (has_mfa * 0.4 + has_ip * 0.3 + has_time * 0.2 + int(bool(all_conditions)) * 0.1)

    # compliance violations (CIS + NIST)
    v = 0
    if any(\'*\' in a for a in all_actions): v += 1
    if wc_res:                             v += 1
    if not all_conditions:                 v += 1
    if esc_match:                          v += 2
    if any(a in [\'*\',\'iam:*\'] for a in all_actions): v += 2

    # approximations for graph features (cannot compute without graph)
    btw_centrality  = min(len(dang_match) / 10.0, 1.0)
    pg_rank         = min((len(esc_match) * 2 + len(dang_match)) / 20.0, 1.0)
    clust_coef      = act_div
    ego_density     = len(dang_match) / n
    subgraph_mod    = 0.5 - (perm_overlap * 0.3)
    cross_acct      = 0

    # rollback risk (createpolicyversion + setdefaultpolicyversion)
    rollback_risk   = int(any(\'iam:createpolicyversion\' in a or \'iam:setdefaultpolicyversion\' in a
                              for a in all_actions))
    max_hist_risk   = esc_score
    unused_ratio    = max(0.0, 1.0 - (len(dang_match) / n))
    policy_ver_cnt  = 1

    return {
        \'betweenness_centrality\':         btw_centrality,
        \'pagerank\':                        pg_rank,
        \'clustering_coefficient\':          clust_coef,
        \'ego_network_density\':             ego_density,
        \'shortest_path_to_admin\':          float(hop),
        \'attachment_count\':                1.0,
        \'service_count\':                   float(len(services)),
        \'resource_count\':                  float(len(all_resources)),
        \'cross_account_edge_count\':        float(cross_acct),
        \'subgraph_modularity\':             subgraph_mod,
        \'wildcard_entropy\':                wc_ent,
        \'specificity_score\':               spec,
        \'dangerous_action_count\':          float(len(dang_match)),
        \'has_wildcard_action\':             float(int(bool(wc_actions))),
        \'has_wildcard_resource\':           float(int(bool(wc_res))),
        \'service_wildcard_count\':          float(len(svc_wc)),
        \'action_diversity\':                act_div,
        \'resource_arn_specificity\':        arn_spec,
        \'permission_overlap_score\':        perm_overlap,
        \'cross_service_permission_chains\': float(cross_chains),
        \'escalation_path_count\':           float(esc_path_count),
        \'min_escalation_path_length\':      float(min_esc_len),
        \'escalation_techniques_enabled\':   float(esc_techniques),
        \'passrole_chain_exists\':           float(has_passrole),
        \'createpolicyversion_exists\':      float(has_cpv),
        \'attachuserpolicy_exists\':         float(has_aup),
        \'iam_write_permission_count\':      float(len(iam_write)),
        \'privilege_escalation_risk_score\': esc_score,
        \'has_mfa_condition\':               float(has_mfa),
        \'has_ip_restriction\':              float(has_ip),
        \'has_time_restriction\':            float(has_time),
        \'condition_protection_score\':      cond_score,
        \'is_bounded\':                      float(is_bounded),
        \'policy_version_count\':            float(policy_ver_cnt),
        \'max_historical_risk\':             max_hist_risk,
        \'rollback_risk_score\':             float(rollback_risk),
        \'unused_permission_ratio\':         unused_ratio,
        \'compliance_violation_count\':      float(v),
    }
'''

# replace the entire old function
old_start = txt.find('def extract_features_from_policy(')
old_end   = txt.find('\ndef ', old_start + 10)
txt = txt[:old_start] + new_func + '\n' + txt[old_end:]
open('src/api_final.py', 'w', encoding='utf-8').write(txt)
print('Done')

baserules = """
    num = ~r"[0-9.]+"
    sym = ~r"[a-zA-Z][a-zA-Z0-9_]*"
    sep = ws? "," ws?
    ws = ~r"[ \\t\\n\\r]*"
"""


atomrules = """
    atom = sym "(" terms ")"

    terms = sym? (sep sym)*
"""


# Disjunction clause
formularules = """
    formula = exist? cnf

    exist = "Exist" ws terms ws

    cnf = "("? clause (and_sep clause)* ")"?

    clause = "("? literal (or_sep literal)* ")"?

    and_sep = ws? and ws?
    and = "^"
    or_sep = ws? or ws?
    or = "v"

    literal = neg? atom
    neg = ~r"[!~]"
""" + atomrules

domainrules = """
    domain = sym ws? "=" ws? (domain_spec / domain_slice / num)?
    domain_spec = "{" sym (sep sym)* "}"
    domain_slice = "{" num sep "..." sep num "}"
"""

constraint_rules = """
    constraints = tree? ws* ccs?

    tree = "Tree[" sym + "]"
    ccs = cc (ws cc)*
    cc = "|" sym "|" ws? "=" ws? num
"""

mlnrules = """
    mln = ws? domain (ws? weighted_formula ws?)+

    weighted_formula = (num ws)? formula hard?
    hard = "."
""" + domainrules + formularules + baserules

mln_with_constraint_rules = """
    mln_with_constraint = mln ws* constraints ws*
""" + constraint_rules + mlnrules

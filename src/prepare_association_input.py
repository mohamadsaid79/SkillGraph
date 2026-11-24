import numpy as np
from Orange.data import Table, Domain, DiscreteVariable

out_data = None
out_learner = None
out_classifier = None
out_object = None

if in_data is not None:
    attributes = in_data.domain.attributes

    # Each skill has only one item: present.
    skill_variables = [
        DiscreteVariable(var.name, values=("Present",))
        for var in attributes
    ]

    matrix = np.full(
        (len(in_data), len(attributes)), np.nan
    )

    for index, var in enumerate(attributes):
        if not var.is_discrete or "1" not in var.values:
            raise ValueError(
                f"Expected a binary skill column: {var.name}"
            )

        present_code = var.values.index("1")
        detected = in_data.get_column(var) == present_code
        matrix[detected, index] = 0

    domain = Domain(
        skill_variables,
        metas=in_data.domain.metas
    )

    out_data = Table.from_numpy(
        domain,
        matrix,
        metas=in_data.metas.copy()
    )

    print("Transactions:", len(out_data))
    print("Skill features:", len(domain.attributes))
    print("Ready for presence-only association rules.")
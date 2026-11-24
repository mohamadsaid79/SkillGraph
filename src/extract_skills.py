import re
import html
import numpy as np
from Orange.data import Table, Domain, DiscreteVariable, StringVariable

out_data = None
out_learner = None
out_classifier = None
out_object = None

if in_data is None:
    print("Connect Unique to Python Script first.")
else:
    # Skill names and alternative spellings.
    skill_patterns = {
        "Python": r"\bpython\b",
        "Java": r"\bjava\b",
        "JavaScript": r"\bjava\s*script\b",
        "TypeScript": r"\btypescript\b",
        "C++": r"(?<!\w)c\+\+(?!\w)",
        "C#": r"(?<!\w)c#(?!\w)|\bc\s*sharp\b",
        "PHP": r"\bphp\b",
        "SQL": r"\bsql\b",
        "MySQL": r"\bmysql\b",
        "PostgreSQL": r"\bpostgres(?:ql)?\b",
        "MongoDB": r"\bmongo\s*db\b",
        "HTML": r"\bhtml5?\b",
        "CSS": r"\bcss3?\b",
        "React": r"\breact(?:\.?js)?\b",
        "Angular": r"\bangular(?:\.?js)?\b",
        "Node.js": r"\bnode[.\s]*js\b",
        "Django": r"\bdjango\b",
        "Laravel": r"\blaravel\b",
        ".NET": r"(?<!\w)\.net\b|\bdot\s*net\b",
        "Linux": r"\blinux\b",
        "Git": r"\bgit\b",
        "Docker": r"\bdocker\b",
        "Kubernetes": r"\bkubernetes\b|\bk8s\b",
        "AWS": r"\baws\b|\bamazon web services\b",
        "Azure": r"\bazure\b",
        "Jenkins": r"\bjenkins\b",
        "Ansible": r"\bansible\b",
        "Terraform": r"\bterraform\b",
        "Hadoop": r"\bhadoop\b",
        "Spark": r"\b(?:apache\s+)?spark\b",
    }

    names = list(skill_patterns)
    patterns = [
        re.compile(skill_patterns[name], re.IGNORECASE)
        for name in names
    ]

    descriptions = in_data.get_column("jobdescription")
    matrix = np.zeros((len(in_data), len(names)), dtype=float)
    extracted = []

    for i, description in enumerate(descriptions):
        text = description if isinstance(description, str) else ""
        text = html.unescape(text)

        # Exclude the company profile to reduce unrelated mentions.
        text = re.split(
            r"\bcompany\s+profile\s*:",
            text,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0]

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        found = []
        for j, pattern in enumerate(patterns):
            if pattern.search(text):
                matrix[i, j] = 1
                found.append(names[j])

        extracted.append("; ".join(found))

    # Keep the original metadata and add a readable skills list.
    variables = [
        DiscreteVariable(name, values=("0", "1"))
        for name in names
    ]

    meta_variables = list(in_data.domain.metas)
    meta_variables.append(StringVariable("extracted_skills"))

    metas = np.column_stack((
        in_data.metas,
        np.asarray(extracted, dtype=object)
    ))

    domain = Domain(variables, metas=meta_variables)
    out_data = Table.from_numpy(domain, matrix, metas=metas)

    print("Input rows:", len(in_data))
    print("Output rows:", len(out_data))
    print("Skill features:", len(names))
    print("Rows with detected skills:", int((matrix.sum(axis=1) > 0).sum()))
    print("Rows without detected skills:", int((matrix.sum(axis=1) == 0).sum()))

    print("\nSkill mention counts:")
    for name, count in sorted(
        zip(names, matrix.sum(axis=0)),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{name}: {int(count)}")
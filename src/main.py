from pathlib import Path

import pandas as pd
import mlxtend
import sklearn
import sqlalchemy
import pymysql
import dotenv
import matplotlib

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

(output_dir / "environment-check.txt").write_text(
    "SkillGraph environment is ready!\n",
    encoding="utf-8",
)

print("SkillGraph environment is ready!")
print("Result saved to outputs/environment-check.txt")
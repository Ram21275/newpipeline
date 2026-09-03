# newpipeline

This repository is set up for a simple local-development → Git repository → Kaggle testing loop.

## Daily workflow

From your computer, make changes and run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
pytest -q
git add .
git commit -m "Describe the change"
git push origin main
```

Keep application code in `src/newpipeline/`, tests in `tests/`, and add Python packages to `requirements.txt`. Do not commit secrets, raw data, or generated outputs; they are ignored by default.

## Run the latest code in Kaggle

1. Create a Kaggle Notebook and enable **Internet** in its settings. This is required for `git clone` and `pip install` from a repository.
2. Add this first cell, replacing the repository URL:

```python
!git clone https://github.com/OWNER/REPOSITORY.git /kaggle/working/newpipeline
%cd /kaggle/working/newpipeline
!pip install -r requirements.txt
```

3. Import and test your code in later cells:

```python
import sys
sys.path.insert(0, "/kaggle/working/newpipeline/src")

from newpipeline import __doc__
print(__doc__)

!pytest -q
```

Each Kaggle session starts fresh, so rerun the clone/install cell after pushing changes locally. To pull a newer revision in the same session, run:

```python
%cd /kaggle/working/newpipeline
!git pull --ff-only origin main
!pip install -r requirements.txt
```

## Private repositories

For a private GitHub repository, Kaggle needs credentials. Store a GitHub fine-grained personal access token in Kaggle's **Add-ons → Secrets** (for example `GITHUB_TOKEN`) and enable it for the notebook. Then clone without putting the token in notebook source or Git history:

```python
import os

token = os.environ["GITHUB_TOKEN"]
repo = "OWNER/REPOSITORY"
!git clone "https://x-access-token:{token}@github.com/{repo}.git" /kaggle/working/newpipeline
```

Use a token limited to that repository with read-only **Contents** permission where possible. Do not print the token or commit it to this repository.

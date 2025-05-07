#!/usr/bin/env python3
"""This script uses the Github API to get the ID of the latest 
successful Documentation run on the main branch.

It then prints the ID"""

import requests

# See here
actions_endpoint = "https://api.github.com/repos/lab-cosmo/atomistic-cookbook/actions"
doc_runs_endpoint = actions_endpoint + "/workflows/docs.yml/runs"

runs = requests.get(doc_runs_endpoint, params={
    "branch": "main", 
    "per_page": 1, 
    "status": "success", 
    "exclude_pull_requests": True
})

latest_successful_run = runs.json()["workflow_runs"][0]

print(latest_successful_run["id"])

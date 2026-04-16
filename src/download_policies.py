# download_policies.py
import requests
import json
import os

# AWS Sample Policies Repository
SAMPLE_POLICIES_URLS = [
    "https://raw.githubusercontent.com/awsdocs/iam-user-guide/main/doc_source/example_policies.json",
    "https://github.com/aws-samples/aws-iam-policies"
]

# CloudGoat vulnerable scenarios
CLOUDGOAT_URL = "https://github.com/RhinoSecurityLabs/cloudgoat"

def download_aws_sample_policies():
    """Download AWS official example policies"""
    os.makedirs('data/raw_policies', exist_ok=True)
    
    # AWS managed policies (publicly available)
    managed_policies = [
        "AdministratorAccess",
        "PowerUserAccess",
        "ReadOnlyAccess",
        "SecurityAudit",
        "ViewOnlyAccess"
    ]
    
    # These are documented in AWS docs, scrape or manually collect
    # For research, you can use AWS Policy Generator examples
    
def clone_cloudgoat():
    """Clone vulnerable-by-design scenarios"""
    os.system("git clone https://github.com/RhinoSecurityLabs/cloudgoat data/cloudgoat")
    
def extract_policies_from_cloudgoat():
    """Parse Terraform configs to extract IAM policies"""
    import glob
    
    terraform_files = glob.glob("data/cloudgoat/**/*.tf", recursive=True)
    policies = []
    
    for tf_file in terraform_files:
        with open(tf_file, 'r') as f:
            content = f.read()
            # Extract policy blocks
            # This requires parsing HCL (Terraform format)
            # Use python-hcl2 library
            
    return policies

if __name__ == "__main__":
    download_aws_sample_policies()
    clone_cloudgoat()

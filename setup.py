from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readline()
        requirements = [req.replace("\n","")for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
        
    return requirements

setup(
    name = "Market_Pulse_Analyzing_Popularity_Patterns_for_Apparel_Brands",
    version = "0.0.1",
    author = "arjundeshmukh",
    author_email="arjundeshmukh757575@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)

     





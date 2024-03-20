from setuptools import find_packages,setup

hyphen_e="-e."

def get_requirements(file_path):
  with open(file_path,'r') as f:
    requirements=[]
    requirements=f.readlines()
    requirements=[item.replace("\n","") for item in requirements]

    
  if hyphen_e in requirements:
    requirements.remove(hyphen_e)

  return requirements


setup(
  name="Loan-Approval-Prediction",
  author='ankitsrivastava',
  author_email='ankitsrivastav2202@gmail.com',
  packages=find_packages(),
  install_require=get_requirements('requirements.txt')
)
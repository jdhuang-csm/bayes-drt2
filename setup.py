from setuptools import setup

with open("README.md", "r",encoding="utf8") as fh:
    long_description = fh.read()

setup(name='bayes-drt2',
	version='0.2',
	description='A Python package for hierarchical Bayesian inversion of electrochemical impedance data',
	long_description=long_description,
    long_description_content_type="text/markdown",
	url='https://github.com/jdhuang-csm/bayes-drt2',
	author='Jake Huang',
	author_email='jdhuang@mines.edu',
	license='BSD 3-clause',
	packages=['bayes_drt2'],
	install_requires=[
		'numpy',
		'scipy',
		'pandas',
		'cvxopt',
		'cmdstanpy',
		'matplotlib'
		],
	include_package_data=True
	)
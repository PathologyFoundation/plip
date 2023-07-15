from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='plip',
    version='0.2.1',
    description='',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    author='PLIP Authors',
    author_email='',
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=[
        'plip',
    ],
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False
    )
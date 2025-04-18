# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""trl is an open library for RL with transformer models.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

0. Prerequisites:
   - Dependencies:
     - twine: "pip install twine"
   - Create an account in (and join the 'trl' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/

1. Change the version in:
   - __init__.py
   - setup.py

2. Commit these changes: "git commit -m 'Release: VERSION'"

3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for pypi'"
   Push the tag to remote: git push --tags origin main

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   First, delete any "build" directory that may exist from previous builds.

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv/notebook by running:
   pip install huggingface_hub fsspec aiohttp
   pip install -U tqdm
   pip install -i https://testpypi.python.org/pypi evaluate

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Fill release notes in the tag in github once everything is looking hunky-dory.

8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
   Then push the change with a message 'set dev version'
"""

import os

from setuptools import find_packages, setup


__version__ = "0.12.0.dev0"  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)

REQUIRED_PKGS = [
    "torch>=1.4.0",
    "transformers>=4.46.0",
    "numpy>=1.18.2;platform_system!='Windows'",
    "numpy<2;platform_system=='Windows'",
    "accelerate",
    "datasets",
    "tyro>=0.5.11",
]
EXTRAS = {
    "test": [
        "parameterized",
        "peft>=0.8.0",
        "pytest",
        "pytest-xdist",
        "pytest-cov",
        "pytest-xdist",
        "scikit-learn",
        "Pillow",
        "pytest-rerunfailures",
        "llm-blender>=0.0.2",
    ],
    "peft": ["peft>=0.8.0"],
    "liger": ["liger-kernel>=0.2.1"],
    "diffusers": ["diffusers>=0.18.0"],
    "deepspeed": ["deepspeed>=0.14.4"],
    "quantization": ["bitsandbytes<=0.41.1"],
    "llm_judge": ["openai>=1.23.2", "llm-blender>=0.0.2"],
}
EXTRAS["dev"] = []
for reqs in EXTRAS.values():
    EXTRAS["dev"].extend(reqs)

try:
    file_path = os.path.dirname(os.path.abspath(__file__))
    os.symlink(os.path.join(file_path, "examples/scripts"), os.path.join(file_path, "trl/commands/scripts"))

    setup(
        name="trl",
        license="Apache 2.0",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        url="https://github.com/huggingface/trl",
        entry_points={
            "console_scripts": ["trl=trl.commands.cli:main"],
        },
        include_package_data=True,
        package_data={"trl": ["commands/scripts/config/*", "commands/scripts/*", "templates/*.md"]},
        packages=find_packages(exclude={"tests"}),
        install_requires=REQUIRED_PKGS,
        extras_require=EXTRAS,
        python_requires=">=3.7",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        zip_safe=False,
        version=__version__,
        description="Train transformer language models with reinforcement learning.",
        keywords="ppo, transformers, huggingface, gpt2, language modeling, rlhf",
        author="Leandro von Werra",
        author_email="leandro.vonwerra@gmail.com",
    )
finally:
    os.unlink(os.path.join(file_path, "trl/commands/scripts"))

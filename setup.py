from setuptools import setup, find_packages

setup(name="rewards-from-language",
      version="0.1",
      author="Jessy Lin",
      author_email="jessy_lin@berkeley.edu",
      packages=[
          "rewards",
          "rewards.models",
          "rewards.posterior_models",
      ],
      install_requires=[
        "pyro-ppl",
        "transformers",
        "torch",
        "einops",
        "jsonlines",
        "wandb",
        ]
      )


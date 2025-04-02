from setuptools import setup, find_namespace_packages

setup(
    name="backend",
    version="1.0.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.95.0",
        "uvicorn>=0.21.1",
        "numpy>=1.22.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.5",
        "python-dotenv>=1.0.0",
        "pydantic-settings>=2.0.0",
    ],
    python_requires=">=3.8",
) 
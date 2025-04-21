DS440 – Pattern Generator

This project helps users generate and retrieve sewing patterns based on clothing descriptions using a model with Retrieval-Augmented Generation (RAG).

Getting Started

1. Open Anaconda Prompt

Make sure Anaconda is installed. Open the Anaconda Prompt to begin setup.

2. Create and Activate Environment

Navigate to the project folder where requirements.txt is located, then run:

conda create --name garment_env --file requirements.txt
conda activate garment_env

If requirements.txt was created using pip freeze instead, use:

pip install -r requirements.txt

3. Run the Script

After activating the environment, run:

python LLM_RAG.py

This starts the pattern retrieval system.

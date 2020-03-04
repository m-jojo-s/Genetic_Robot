1- Create a new environment called ai_env:

	>>> conda create -n ai_env python=3.6

   For more information on environments, visit: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html. You can also create a environment using anaconda.

2- Activate that environment.
	Windows: >>> activate ai_env
	Linux and macOS: >>> source activate ai_env
	For conda 4.6 and later versions: >>> conda activate ai_env

3- Install packages specified in `requirements.txt` using the command below:

	>>> pip install -r requirements.txt

4- Then launch notebook from the same environment (ai_env)
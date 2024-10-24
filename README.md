# MLE Case Study

Hello Covestro team, this repo contains the code for all the three tasks for the given MLE Case study.

## Use of repo

1. Install Requirements
Once cloned, please run the `pip install -r requirements.txt` to install the necessary packages.


I have also included the `requirements-dev.txt` - this is only to isolate the development/contributing dependencies from
the main requirements file.

2. TASK 1 - Run The Training Pipeline Code
The `src/configs/config.yaml` is the configuration file that contains the parameters for the training pipeline.
This also includes the path to the data file.
Running `python src/scripts/train.py` will use the existing data, performs the training and saves the model in the
`saved_models' directory.

Any changes to data source, model parameters, model save path etc. can be made in the `config.yaml` file.

3. TASK 2 -Model endpoint

The model endpoint is created using Flask. The model is loaded from the saved model file and the prediction is made
using the input data.
The model endpoint can be started locally using `python src/endpoints/model_endpoint.py`.

4. TASK 3 - Build a CI/CD pipeline

The CI/CD pipeline is created using GitHub Actions. The workflow files for the train pipeline and the inference pipeline (model endpoint)
are located in the `.github/workflows` directory.

I have created a blank PR to show the CI/CD pipeline in action.

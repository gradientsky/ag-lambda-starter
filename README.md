# Deploying AutoGluon models with serverless templates

## Creating a base project

To start the project, please follow the setup steps of the tutorial: 
[Deploying machine learning models with serverless templates](https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/).

To deploy AutoGluon, the following adjustments would be required:
- the trained model is expected to be in `ag_models` directory. The trained model can be produced and saved to the necessary location vi [`training.ipynd`](./training.ipynb) notebook.
- [`Dockerfile`](./app/Dockerfile) to package AutoGluon runtimes and model files
- Modify serving [`app.py`](./app/app.py) script to use AutoGluon

When building a docker container it's size can be reduced using the following optimizations: 
- use CPU versions of `pytorch`; if the models to be deployed don't use `pytorch`, then don't install it.
- install only the AutoGluon sub-modules required for inference - specifically `autogluon.tabular[all]` will deploy only all tabular models 
without `text` and `vision` modules and their extra dependencies. This instruction can be further narrowed down to a combination of 
the following options are: `lightgbm`, `catboost`, `xgboost`, `fastai` and `skex`.

## Reducing the model size to minimize AWS Lambda startup times

When the Lambda service receives a request to run a function via the Lambda API, the service first prepares an execution environment. During this step, the service 
downloads the code for the function, which is stored in Amazon Elastic Container Registry. It then creates an environment with the memory, runtime, and configuration 
specified. Once complete, Lambda runs any initialization code outside of the event handler before finally running the handler code. The steps of setting up the 
environment and the code are frequently referred to as a "cold start".

After the execution completes, the execution environment is frozen. To improve resource management and performance, the Lambda service retains the execution environment 
for a non-deterministic period of time. During this time, if another request arrives for the same function, the service may reuse the environment. This second request 
typically finishes more quickly, since the execution environment already exists and it’s not necessary to download the code and run the initialization code. 
This is called a "warm start".

Because AutoGluon containers are larger than a typical Lambda container, it might take some time (60+ seconds) to perform steps required for a "cold start".  
This could be limiting factor when used with latency-sensitive applications. To reduce start up times with AWS Lambda it is important to reduce model size to a minimum. 
This can be done by applying deployment-optimized presets as described in section "Faster presets or hyperparameters" of :ref:`sec_tabularadvanced`:

```python
presets = ['good_quality_faster_inference_only_refit', 'optimize_for_deployment']
```

If the cold boot latency cannot be tolerated, it is recommended to reserve concurrent capacity as described in this article:
[Managing Lambda reserved concurrency](https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html).

More details on the lambda performance optimizations can be found in the following article: 
[Operating Lambda: Performance optimization](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/)


# Appendix - SAM Tools

This project contains source code and supporting files for a serverless application for classifying handwritten digits using a Machine Learning model in [scikit-learn](https://scikit-learn.org/). It includes the following files and folders:

- app/app.py - Code for the application's Lambda function including the code for ML inferencing.
- app/Dockerfile - The Dockerfile to build the container image.
- app/model - A simple scikit-learn model for classifying handwritten digits trained against the MNIST dataset.
- app/requirements.txt - The pip requirements to be installed during the container build.
- events - Invocation events that you can use to invoke the function.
- template.yaml - A template that defines the application's AWS resources.
- training.ipynb - A jupyter notebook to show the training process for the sample model at `app/model`.

The application uses several AWS resources, including Lambda functions and an API Gateway API. These resources are defined in the `template.yaml` file in this project. You can update the template to add AWS resources through the same deployment process that updates your application code.

While this template does not use any auth, you will almost certainly want to use auth in order to productionize. Please follow [these instructions](https://github.com/aws/serverless-application-model/blob/master/versions/2016-10-31.md#function-auth-object) to set up auth with SAM.

## Deploy the sample application

The Serverless Application Model Command Line Interface (SAM CLI) is an extension of the AWS CLI that adds functionality for building and testing Lambda applications. It uses Docker to run your functions in an Amazon Linux environment that matches Lambda. It can also emulate your application's build environment and API.

To use the SAM CLI, you need the following tools.

* SAM CLI - [Install the SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html)
* Docker - [Install Docker community edition](https://hub.docker.com/search/?type=edition&offering=community)

You may need the following for local testing.
* [Python 3 installed](https://www.python.org/downloads/)

To build and deploy your application for the first time, run the following in your shell:

```bash
sam build
sam deploy --guided
```

The first command will build a docker image from a Dockerfile and then copy the source of your application inside the Docker image. The second command will package and deploy your application to AWS, with a series of prompts:

* **Stack Name**: The name of the stack to deploy to CloudFormation. This should be unique to your account and region, and a good starting point would be something matching your project name.
* **AWS Region**: The AWS region you want to deploy your app to.
* **Confirm changes before deploy**: If set to yes, any change sets will be shown to you before execution for manual review. If set to no, the AWS SAM CLI will automatically deploy application changes.
* **Allow SAM CLI IAM role creation**: Many AWS SAM templates, including this example, create AWS IAM roles required for the AWS Lambda function(s) included to access AWS services. By default, these are scoped down to minimum required permissions. To deploy an AWS CloudFormation stack which creates or modifies IAM roles, the `CAPABILITY_IAM` value for `capabilities` must be provided. If permission isn't provided through this prompt, to deploy this example you must explicitly pass `--capabilities CAPABILITY_IAM` to the `sam deploy` command.
* **Save arguments to samconfig.toml**: If set to yes, your choices will be saved to a configuration file inside the project, so that in the future you can just re-run `sam deploy` without parameters to deploy changes to your application.

You can find your API Gateway Endpoint URL in the output values displayed after deployment.

## Use the SAM CLI to build and test locally

Build your application with the `sam build` command.

```bash
ag-lambda$ sam build
```

The SAM CLI builds a docker image from a Dockerfile and then installs dependencies defined in `app/requirements.txt` inside the docker image. The processed template file is saved in the `.aws-sam/build` folder.

Test a single function by invoking it directly with a test event. An event is a JSON document that represents the input that the function receives from the event source. Test events are included in the `events` folder in this project.

Run functions locally and invoke them with the `sam local invoke` command.

```bash
ag-lambda$ sam local invoke InferenceFunction --event events/event.json
```

The SAM CLI can also emulate your application's API. Use the `sam local start-api` to run the API locally on port 3000.

```bash
ag-lambda$ sam local start-api
ag-lambda$ curl http://localhost:3000/classify_digit
```

The SAM CLI reads the application template to determine the API's routes and the functions that they invoke. The `Events` property on each function's definition includes the route and method for each path.

```yaml
      Events:
        Inference:
          Type: Api
          Properties:
            Path: /classify_digit
            Method: post
```

## Add a resource to your application
The application template uses AWS Serverless Application Model (AWS SAM) to define application resources. AWS SAM is an extension of AWS CloudFormation with a simpler syntax for configuring common serverless application resources such as functions, triggers, and APIs. For resources not included in [the SAM specification](https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md), you can use standard [AWS CloudFormation](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html) resource types.

## Fetch, tail, and filter Lambda function logs

To simplify troubleshooting, SAM CLI has a command called `sam logs`. `sam logs` lets you fetch logs generated by your deployed Lambda function from the command line. In addition to printing the logs on the terminal, this command has several nifty features to help you quickly find the bug.

`NOTE`: This command works for all AWS Lambda functions; not just the ones you deploy using SAM.

```bash
ag-lambda$ sam logs -n InferenceFunction --stack-name ag-lambda --tail
```

You can find more information and examples about filtering Lambda function logs in the [SAM CLI Documentation](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-logging.html).

## Cleanup

To delete the sample application that you created, use the AWS CLI. Assuming you used your project name for the stack name, you can run the following:

```bash
aws cloudformation delete-stack --stack-name ag-lambda
```

## Resources

See the [AWS SAM developer guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html) for an introduction to SAM specification, the SAM CLI, and serverless application concepts.

Next, you can use AWS Serverless Application Repository to deploy ready to use Apps that go beyond hello world samples and learn how authors developed their applications: [AWS Serverless Application Repository main page](https://aws.amazon.com/serverless/serverlessrepo/)

# Session 4

Build a Deep Learning app. We will train a model that classifies text reviews between positive and negative, and use it in a web app that predicts the sentiment of any review the user writes.
Then, we will deploy the app to the cloud using Google Cloud Run.

## Dataset

We will use the Yelp Review polarity dataset ([link](https://www.kaggle.com/datasets/irustandi/yelp-review-polarity), [direct download](https://www.kaggle.com/api/v1/datasets/download/irustandi/yelp-review-polarity)).

The dataset has two labels. "1" corresponds to negative reviews, and "2" to positive reviews.

You don't need it for Task 1, so you can start the download now and keep going. It is a 339 MB zip.

## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name mlops-session4 python=3.10
```
Then, activate the environment
```bash
conda activate mlops-session4
```
and install the dependencies
```bash
pip install -r session-4/requirements.txt
```

**Note:** it is important to create a new conda environment, and not re-use previous ones from other sessions.

### With venv
Create a virtual environment by running

```bash
python -m venv .venv
source .venv/bin/activate
```
and install the dependencies
```bash
pip install -r session-4/requirements.txt
```

## Tasks

### Task 1: Build the app

Everything web related in the `app/` folder is already written. You only have to complete the two parts that use the model: look for the `TODO` comments in `app/main.py`.

An untrained checkpoint (`app/state_dict.pt`) is already provided, so you can test the app right away. Its predictions will be random, but you will see the app working.

1. Complete the two `TODO`s in `app/main.py`. The model (`app/model.py`) is already written, read it to understand its inputs and outputs.
2. Run the app:
```bash
cd session-4/app
python main.py
```
3. Go to http://localhost:8080 and submit a review. Since the model is untrained, the predictions are random. That's expected!

### Task 2: Train the model

Now let's train a real model, so the app gives meaningful predictions.

1. Download the [dataset](https://www.kaggle.com/api/v1/datasets/download/irustandi/yelp-review-polarity) and extract it at the root of the repository. You should end up with a `yelp_review_polarity_csv/` folder next to `session-4/`, with `train.csv` and `test.csv` inside. (The zip contains a second copy of the files inside another `yelp_review_polarity_csv/` folder, you can ignore or delete it.)
2. Run the training:
```bash
python session-4/train.py
```
3. The training script is complete, but **it has two bugs**. Run it, read what happens, and fix them one by one:
   - The first bug crashes the script with an error.
   - The second bug does not crash anything: the script runs, prints numbers, and finishes. But look closely at the numbers.

   When both bugs are fixed, each epoch takes less than a minute on a laptop, and you should see a validation accuracy around 90%.

4. The script saves the trained model to `app/state_dict.pt`, replacing the untrained one.
5. Run the app again and see the difference: the predictions should now make sense!

### Task 3: Deploy to Google Cloud Run (optional)

Deploy the app to the cloud, so anyone with the URL can use it.

To do it, we will use **Cloud Shell**: a terminal in your browser, inside the Google Cloud console. It already has `gcloud` (the Google Cloud command line tool) installed and logged in with your account, so you don't have to install yet another tool in your computer, and the commands are the same for everyone (Windows, Mac and Linux).

1. Open the [Google Cloud console](https://console.cloud.google.com/), select your project and click the *Activate Cloud Shell* button (the terminal icon at the top right). A terminal opens at the bottom of the page.
2. Get your code into Cloud Shell: click *Open Editor* (top right of the terminal). In the editor's file tree, drag and drop your `session-4` folder from your computer into the home folder. Don't drag the whole repository: it contains the dataset (877 MB). Your trained `state_dict.pt` (110 MB) is inside `session-4/app`, so it goes with it; if you want the upload to be faster, delete it first and deploy with the untrained checkpoint from the repository.
3. Back in the terminal, go to the app folder:
```bash
cd session-4/app
```
4. Give Cloud Build the permissions it needs. You only need to do this once per project:
```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder"
```
5. Deploy, from the `app/` folder:
```bash
gcloud run deploy sentiment-analysis \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi
```
   The first time, it asks to enable some APIs: answer `y`. It then builds a container with your app in the cloud and starts it. The build takes about 3 minutes, plus the time to upload your files.

   If it fails with `PERMISSION_DENIED` right after the permissions command of step 4, wait a minute and run it again: the permissions take a moment to become active.
6. Once deployed, it prints a URL like `https://sentiment-analysis-XXXXX.europe-west1.run.app`. Open it and try it. Send it to a friend!

If you already have `gcloud` installed in your computer, the same commands work from your laptop, from the `session-4/app` folder.

#### Cleanup

Cloud Run only charges when someone uses the app, and the first hours per month are free, but when you are done you can delete the service:

```bash
gcloud run services delete sentiment-analysis --region europe-west1
```

### Introduction
Private banks face a crucial challenge in ensuring that client onboarding processes are efficient, secure, and compliant with regulatory requirements. Manual verification of client data can be time-consuming and prone to errors, leading to delayed or incorrect onboarding decisions.

### Challenge Objective
In this 24-hour datathon challenge, participants will develop an automated solution to verify the consistency of client data and predict whether a client should be accepted or rejected based on their profile information.

### Dataset Description
Participants will be provided with a training dataset consisting of 10,000 client profiles, each comprising five JSON documents:

* passport.json: Client identification details
* client_profile.json: Demographic and personal information
* account_form.json: Account opening details
* client_description.json: KYC description of a client
* label.json: Binary label indicating whether the client data is consistent and acceptable (Accept) or not (Reject)

### Task
Using the provided training dataset, participants must develop a machine learning model or algorithm that can automatically verify the consistency of client data and predict the corresponding label.

### Evaluation Dataset
After the development period, participants will receive a separate evaluation dataset consisting of 1,000 client profiles, each with four JSON documents (passport, client_profile, account_form, and client_description). Their task is to apply their solution to these new client profiles and generate a CSV file with two columns (no header, no index):

* client_id: Unique identifier for each client
* label: Predicted binary label (Accept or Reject) indicating whether the client data is consistent and acceptable

### Deliverables
A single CSV file containing the predicted labels for the 1,000 client profiles in the evaluation dataset   
Codebase with solution   
Presentation describing the approach, methodology, and any insights gained during the challenge (5 minutes)

### Description of Repository
Following files are available in repository:

* datathon.zip - archive with 10,000 clients. Each client is a separate archive, consisting of five files mentioned
  * this archive is split into four parts, named datathon_part[i].zip
* solution.csv - template CSV file to show the shape of expected solution delivered for evaluation set
* evaluation.zip - archive with 1,000 clients. Each client is a separate archive, consisting of four files mentioned
  * this file will be submitted shortly before end of the challenge

### Evaluation of Solution
Solution provided will be evaluated according to following criteria:
* accuracy in provided solution file
* solution design criteria like efficiency, novelty, elegance and applicability within the banking sector
* presentation of the solution

### Hand-in
To evaluate prediction score, classification labels have to be provided in the same format as in the provided example. Name your submission csv file as "<team_name>.csv". Please fork this repository, start developing your project, and ensure your final submission, including your code, is uploaded to your forked repository **before Sunday 12:00pm**, as this will be used to evaluate your solution.

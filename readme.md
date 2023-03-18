# Penguins Classifier
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white)
![scikitlearn](https://img.shields.io/badge/Scikit_Learn-4D97FF?logo=scikit-learn&logoColor=white)
![terminal](https://img.shields.io/badge/Windows%20Terminal-4D4D4D?&logo=Windows%20terminal&logoColor=white)
![vscode](https://img.shields.io/badge/Visual_Studio_Code-0078D4?&logo=visual%20studio%20code&logoColor=white)

This repository contains code for a web application built with streamlit and scikit-learn that predicts the type of an penguins type based on its bill length, bill depth, flipper length, body mass, island, and sex. The app allows users to input values for these parameters using sliders and displays the predicted penguin type and the probability of the prediction.

## Installation
Open command prompt and create new environment
```
conda create -n your_env_name python = (any_version_number > 3.10)
```
Then Activate the newly created environment
```
conda activate your_env_name
```
Clone the repository using `git`
```
git clone https://github.com/Prakashdeveloper03/Penguins-Classifier.git
```
Change to the cloned directory
```
cd <directory_name>
```
Then install all requirement packages for the app
```
pip install -r requirements.txt
```
Then, Run the `app.py` script
```
streamlit run app.py
```
## 📷 Screenshots
![output_image](markdown/interface.png)
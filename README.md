# 🔬 Data Science Project Portfolio
Welcome to my Data Science Project Portfolio! This repository serves as a showcase of my work in various domains of data science, ranging from predictive modeling and statistical analysis to machine learning algorithms. Each project aims to demonstrate my ability to tackle real-world problems, extract insights from data, and build robust analytical solutions.

## 🎯 About This Repository
This collection of projects highlights my practical experience with data manipulation, exploratory data analysis (EDA), model building, evaluation, and interpretation. The goal is to provide a clear understanding of my thought process, methodologies, and the impact of the data-driven solutions I develop.

Each project folder typically contains:

Jupyter Notebooks (.ipynb) detailing the analysis.
R scripts/files
Relevant datasets (if permissible and small enough).
Markdown files or comments explaining the process and findings.

## 📂 Project Showcase
Here's an overview of the projects included in this repository. Click on the links to explore the detailed notebooks and code for each.

### 📊 Python Projects

#### 📈 Predictive Analytics
Project focused on forecasting future outcomes for an E-Commerce company's operational improvement utilising various supervised and unsupervised learning techniques.

❓Problems
* Predicting customer purchase behavior to enhance e-commerce operational performance, utilising the "Online Shoppers Purchasing Intention Dataset"(Sakar, C. and Kastro, Y. (2018) ‘Online Shoppers Purchasing Intention Dataset’. UCI Machine Learning Repository. Available at: https://doi.org/10.24432/C5F88Q.).
* Grouping the "Online Shoppers Purchasing Intention Dataset" into meaningful labels for customer segmentation, aiding in enhanced personalisation in customer marketing
* The goal is to find the target customer groups, and to analyse whether a customer will finalise a transaction.

⚙️ Techniques
* 📊 Logistic Regression
* 🌳 Random Forest Classification
* 👥 Cluster Analysis

🪜 Methodology
* Applied Logistic Regression, a statistical method that estimates the probability of a specific event by applying a logistic function to predictor variables.
* Implemented Random Forest Classifier, an effective machine learning technique that employs multiple decision trees to improve accuracy and reduce overfitting, especially in complex datasets. 
* These models were developed to predict customer purchase behavior
* Utilised K-Means and K-Modes Clustering to group similar objects based on their characteristics, aiding in recognising patterns within the data. The Elbow method was used to determine the optimal number of clusters and avoid overfitting.

💡 Key Insights/Outcome
* The Random Forest Classification model outperformed the Logistic Regression model with a test accuracy of 89%.
* Feature Importance analysis identified key drivers like "Bounce Rates" and "PageValues".
* The study shows the model's potential to facilitate data-driven decision-making, enabling proactive problem-solving and resource optimisation.

💻 Technologies: Python, scikit-learn, pandas, numpy, matplotlib, seaborn.

➡️ Explore Project: [Here](https://github.com/comodu20/My-Work_ipynb/blob/ae26cdf0ae8cded33d784dd985c0b6b7f225c9c3/UEL-DS/Summative%20assessment_mod%204/Predictive%20analysis%20of%20customer%20intention%20with%20web%20browser%20data.ipynb)



#### ⛓️ Markov Chains
Projects exploring stochastic processes where future states depend only on the current state, often used in sequential data analysis.

**🚶 Markov Chain for Fast food restaurant**

❓Problems
* Simulating a random walk through a restaurant's menu, where there are three states (meals on the menu)
* Finding the stationary probability distribution
* Computing probability for a specific sequence

🪜 Methodology 
* Constructed a Markov Chain model to calculate transition probabilities between states and predict long-term behavior.

💡 Key Insights/Outcome
* On random walks: Markov chains are used to model stock price movements, where the future price depends only on the current price (memoryless property). Simulating random walks helps in understanding market trends and risk assessment.
* On methodology finding stationary probability distribution:
  - The accuracy of the Monte Carlo simulation apprach increases as number of steps increases
  - Repeated Matrix Multiplication method is fastest
  - The Left Eigen Vector method (similar to output in singular value decomposition (SVD))

💻 Technologies: Python, numpy, pandas.

➡️ Explore Project: [Here](https://github.com/comodu20/My-Work_ipynb/blob/565ec2332b27923eb3a2fa563deb94e2754b39d7/UEL-DS/Markov%20processes/Markov%20chain%20simulation_1.ipynb)


#### 📉 Linear Regression
Projects involving predicting a continuous outcome based on one or more predictor variables.

**🏠 Simple Linear Regression for Salary and Length of Service**

❓Problem
* Simple salary predictor based on years of experience 

🪜 Methodology 
* Assign x and y values
* Split data into training and testing sets
* Build and fit a linear regression model from scikit learn

💡 Key Insights/Outcome
* Good accuracy for predictor.
* This was a simple task to gain familiarity, at the time, with regression models

💻 Technologies: Python, scikit-learn, pandas, numpy, matplotlib, seaborn.

Explore Project: [Here](https://github.com/comodu20/My-Work_ipynb/blob/2e79b50a85eb4003a14cf73500cd03090bcca3b4/UEL-DS/Linear%20Regression/Simple%20Linear%20Regression.ipynb)


### 📊 R Projects

A collection of data analysis and statistical modeling projects implemented using the R programming language. These projects showcase various techniques for data exploration, hypothesis testing, machine learning, and visualisation in R. I have summarised a select few here.

#### 🔄 2-Way ANOVA

❓ Problem 
* Analyzing the effects of two independent categorical variables on a continuous outcome.
* Datasets are Toothgrowth Rdata and randomised data for a book shop and units sold between 3 different genres

🪜 Methodology
* Performed a 2-Way Analysis of Variance to test for significant interactions and main effects  .

💡 Key Insights/Outcome
* Identified significant effects of relevant Factor A, Factor B, and their interaction on the response variable.

💻 Technologies: R (e.g., `aov`, `ggplot2`).

➡️ Explore Project: [Here](https://github.com/comodu20/Data_Science_projects/tree/e53dbb065a0d51af8a8f7c640c620434428e0198/My-Work_R/Exploring_R/2way%20Anova%20practice)

#### 🗺️ Spatial Data Analysis

❓ Problem 
* Exploring and visualizing geographically referenced data to identify spatial patterns.
* From [Camden census data packet](https://data.cdrc.ac.uk/dataset/introduction-spatial-data-analysis-and-visualisation-r/resource/practical-1-introduction-r)

🪜 Methodology
* Utilized R packages for loading, manipulating, and visualizing spatial data

💡 Key Insights/Outcome
* Demonstrated techniques for mapping spatial data and identifying geographic clusters

💻 Technologies: R (e.g., `sf`, `ggmap`, `sp`, `ggplot2`, `leaflet`).

➡️ Explore Project:** [Here](https://github.com/comodu20/Data_Science_projects/blob/e53dbb065a0d51af8a8f7c640c620434428e0198/My-Work_R/Exploring_R/An%20Introduction%20to%20Spatial%20Data%20Analysis%20and%20Visualisation%20in%20R%20_%20CDRC%20Data_files/Practice.R)

#### 🌲 Decision Trees

❓ Problem
* Building a predictive model for classification or regression tasks using a tree-like structure.
* Regression: Predict the target variable (median house value) based on 13 features (independent variables) in the BostonHousing dataset
* Classification: Email spam detection from spam7 Rdataset

🪜 Methodology
* Constructed and pruned decision tree models, visualising decision rules and feature importance.
* Used ROC curve to find a balance between false negatives and false positives

💡 Key Insights/Outcome
* The average number of rooms per dwelling is a highly crucial feature, suggesting that houses with more rooms tend to have higher median values.
* The percentage of the population with lower socioeconomic status is another significant variable, indicating that lower-status neighborhoods are likely to have lower median home values.
* The weighted distance to employment centers is a vital variable, showing that houses closer to major employment centers are likely to have higher median values.

💻 Technologies: R (e.g., `rpart`, `rpart.plot`, `caret`, `tidymodels`).

➡️ Explore Projects: [Here](https://github.com/comodu20/Data_Science_projects/tree/e53dbb065a0d51af8a8f7c640c620434428e0198/My-Work_R/Exploring_R/Decision%20trees)

#### 🧮 Factor Analysis

❓ Problem
* Reducing the dimensionality of a dataset (13 independent variables) by identifying underlying latent factors.

🪜 Methodology
* Performed Exploratory Factor Analysis to uncover relationships between observed variables and a smaller set of unobserved factors.

💡 Key Insights/Outcome
* Identified 3 significant underlying factors explaining the variance in the dataset, simplifying further analysis.
* Assessments here are highly subjective.

💻 Technologies:** R (e.g., `psych`, `foreign`, `RWeka`).

➡️ Explore Project: [Here](https://github.com/comodu20/Data_Science_projects/tree/e53dbb065a0d51af8a8f7c640c620434428e0198/My-Work_R/Exploring_R/Factor%20analysis)


#### 📉 Principal Component Analysis
❓ Problem
* Reducing the dimensionality of a high-dimensional dataset while retaining most of its variance.

🪜 Methodology
* Performed Principal Component Analysis (PCA) to transform variables into a new set of orthogonal components.

💡 Key Insights/Outcome
* I successfully reduced 100 independent variables to 2 principal components, explaining 92+% of the original variance, simplifying subsequent modeling.

💻 Technologies: R (e.g., `prcomp`, `factoextra`, `ggplo2`).

➡️ Explore Project:** ➡️ [Here](https://github.com/comodu20/Data_Science_projects/tree/2763b22c16129325db7813a65ba6b746ac689730/My-Work_R/Exploring_R/PCA%20practice)

#### 😊 Sentiment Analysis

❓ Problem
* Extracting and quantifying sentiment (positive, negative, neutral) from text data.
* Using restaurant_reviews

🪜 Methodology
* Applied Natural Language Processing (NLP) techniques and lexical dictionaries to analyze text and determine sentiment scores

💡 Key Insights/Outcome
* Analysed customer reviews, identifying dominant sentiments towards product features and uncovering areas for improvement.

💻 Technologies: R (e.g., `tidytext`, `textdata`, `dplyr`, `ggplot2`, `syuzhet`).

➡️ Explore Project: [Here](https://github.com/comodu20/Data_Science_projects/tree/2763b22c16129325db7813a65ba6b746ac689730/My-Work_R/Exploring_R/Sentiment%20analysis)



## 💻 Technologies & Libraries
This portfolio primarily leverages the Python ecosystem for data science, including:

Python: Core programming language.
* **Python:** Core programming language.
* **R:** Core programming language for statistical computing and graphics.
* **Pandas / dplyr:** Data manipulation and analysis.
* **NumPy:** Numerical computing.
* **Scikit-learn / caret:** Machine learning algorithms and tools.
* **Matplotlib / Seaborn / ggplot2:** Data visualization.
* **Jupyter Notebooks / R Markdown:** Interactive computing and reproducible reports.
* **Specific R packages:** `sf`, `leaflet`, `psych`, `VIM`, `mice`, `rpart`, `markovchain`, `tidytext`, `tidyverse`, `textdata`, etc.
* **Plotly / Dash**


## ▶️ How to Navigate and Run Projects Locally
To explore these projects on your local machine:

1. Clone the repository:
  ```bash:
  git clone https://github.com/comodu20/Data_Science_projects.git
  cd Data_Science_projects
  ```

2. Create a virtual environment (recommended for Python projects):
    ```bash
    python -m venv venv
    #### On Windows:
    .\venv\Scripts\activate
    #### On macOS/Linux:
    source venv/bin/activate
    ```

3. Install dependencies:
* **For Python projects:** Most projects will rely on standard data science libraries. You can try installing common ones or check individual project folders for a `requirements.txt` file.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

* **For R projects:** Open the R script or R Markdown file in RStudio. You'll typically need to install required packages within R using `install.packages("package_name")` for each package used in the script.

#### Or if a requirements.txt exists:
  ```bash
  pip install -r requirements.txt
  ```
4. **Launch Jupyter Notebook (for Python notebooks):**
    ```bash
    jupyter notebook
    ```
    Or open RStudio and navigate to the R project folders to run `.R` or `.Rmd` files.

5.  Navigate to the respective project folders and open the `.ipynb` (Python) or `.Rmd`/`.R` (R) files to view and run the code.

## 🤝 Connect With Me
I'm always keen to discuss data science, collaborate on projects, or explore new opportunities. Feel free to reach out!

LinkedIn: [Morgan Omodu](https://www.linkedin.com/in/morganomodu/)
Email: omodumorgan@gmail.com
GitHub: [Here](https://github.com/comodu20)

License
This repository is licensed under the MIT License. See the LICENSE file for more details.

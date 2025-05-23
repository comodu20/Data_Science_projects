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

🚶 Markov Chain for Fast food restaurant

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

🏠 Simple Linear Regression for [Your Project Name/Domain]

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

A collection of data analysis and statistical modeling projects implemented using the R programming language. These projects showcase various techniques for data exploration, hypothesis testing, machine learning, and visualisation in R.

#### 🔄 2-Way ANOVA
❓ Problem 
* Analyzing the effects of two independent categorical variables on a continuous outcome.
🪜 Methodology
* Performed a 2-Way Analysis of Variance to test for significant interactions and main effects.
💡 Key Insights/Outcome
* Identified significant effects of Factor A, Factor B, and their interaction on the response variable."]
* **Technologies:** 💻 R (e.g., `aov`, `ggplot2`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/2way-anova/your_script.Rmd or relevant folder]

#### 🗺️ Introduction to Spatial Data Analysis
* **Problem:** ❓ [Briefly describe the problem, e.g., "Exploring and visualizing geographically referenced data to identify spatial patterns."]
* **Methodology:** 🪜 Utilized R packages for loading, manipulating, and visualizing spatial data.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Demonstrated techniques for mapping spatial data and identifying geographic clusters."]
* **Technologies:** 💻 R (e.g., `sf`, `ggplot2`, `leaflet`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/intro-to-spatial-data-analysis/your_script.Rmd or relevant folder]

#### 🎲 Chi-Square Test
* **Problem:** ❓ [Briefly describe the problem, e.g., "Testing for association between two categorical variables."]
* **Methodology:** 🪜 Applied Chi-Square tests (Goodness-of-Fit and Independence) to analyze observed frequencies against expected frequencies.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Determined a statistically significant association between X and Y categorical variables."]
* **Technologies:** 💻 R (e.g., `chisq.test`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/chi-square/your_script.Rmd or relevant folder]

#### 👥 Cluster Analysis
* **Problem:** ❓ [Briefly describe the problem, e.g., "Segmenting a dataset into natural groupings without prior labels."]
* **Methodology:** 🪜 Implemented various clustering algorithms (e.g., K-Means, Hierarchical Clustering) and evaluated optimal cluster numbers.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Identified distinct clusters within the data, revealing underlying patterns for segmentation."]
* **Technologies:** 💻 R (e.g., `stats`, `factoextra`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/cluster-analysis/your_script.Rmd or relevant folder]

#### 📏 Cronbach's Alpha
* **Problem:** ❓ [Briefly describe the problem, e.g., "Assessing the internal consistency reliability of a psychometric scale or questionnaire."]
* **Methodology:** 🪜 Calculated Cronbach's Alpha coefficient to measure how closely related a set of items are as a group.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Determined that the scale demonstrated high internal consistency (Cronbach's Alpha = X)."]
* **Technologies:** 💻 R (e.g., `psych`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/cronbach-alpha/your_script.Rmd or relevant folder]

#### 🌲 Decision Trees
* **Problem:** ❓ [Briefly describe the problem, e.g., "Building a predictive model for classification or regression tasks using a tree-like structure."]
* **Methodology:** 🪜 Constructed and pruned decision tree models, visualizing decision rules and feature importance.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Developed a clear, interpretable model to predict X, with key decision rules based on features Y and Z."]
* **Technologies:** 💻 R (e.g., `rpart`, `rpart.plot`, `caret`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/decision-trees/your_script.Rmd or relevant folder]

#### 🧮 Factor Analysis
* **Problem:** ❓ [Briefly describe the problem, e.g., "Reducing the dimensionality of a dataset by identifying underlying latent factors."]
* **Methodology:** 🪜 Performed Exploratory Factor Analysis to uncover relationships between observed variables and a smaller set of unobserved factors.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Identified 3 significant underlying factors explaining the variance in the dataset, simplifying further analysis."]
* **Technologies:** 💻 R (e.g., `psych`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/factor-analysis/your_script.Rmd or relevant folder]

#### 🗑️ Handling Missing Values
* **Problem:** ❓ [Briefly describe the problem, e.g., "Addressing incomplete data in a dataset to prepare it for analysis."]
* **Methodology:** 🪜 Explored various imputation techniques (e.g., mean, median, mode, regression imputation) and deletion strategies.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Implemented a robust missing value handling strategy that improved model performance by X% and maintained data integrity."]
* **Technologies:** 💻 R (e.g., `VIM`, `mice`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/handling-missing-values/your_script.Rmd or relevant folder]

#### 🔗 Logistic Regression
* **Problem:** ❓ [Briefly describe the problem, e.g., "Predicting a binary outcome (e.g., yes/no, success/failure) based on predictor variables."]
* **Methodology:** 🪜 Built and evaluated Logistic Regression models to estimate the probability of an event occurring.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Developed a model to predict X with Y accuracy, identifying key predictors and their influence."]
* **Technologies:** 💻 R (e.g., `glm`, `caret`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/logistic-regression/your_script.Rmd or relevant folder]

#### 🔗 Markov Chains
* **Problem:** ❓ [Briefly describe the problem, e.g., "Modeling sequential dependencies and transition probabilities in a system."]
* **Methodology:** 🪜 Constructed Markov Chain models to analyze state transitions and long-term probabilities.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Analyzed customer journey paths, identifying common transitions and steady-state probabilities within a sales funnel."]
* **Technologies:** 💻 R (e.g., `markovchain`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/markov-chains/your_script.Rmd or relevant folder]

#### 📉 PCA Practice
* **Problem:** ❓ [Briefly describe the problem, e.g., "Reducing the dimensionality of a high-dimensional dataset while retaining most of its variance."]
* **Methodology:** 🪜 Performed Principal Component Analysis (PCA) to transform variables into a new set of orthogonal components.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Successfully reduced 50 variables to 5 principal components, explaining 90% of the original variance, simplifying subsequent modeling."]
* **Technologies:** 💻 R (e.g., `prcomp`, `factoextra`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/pca-practice/your_script.Rmd or relevant folder]

#### 😊 Sentiment Analysis
* **Problem:** ❓ [Briefly describe the problem, e.g., "Extracting and quantifying sentiment (positive, negative, neutral) from text data."]
* **Methodology:** 🪜 Applied natural language processing (NLP) techniques and lexical dictionaries to analyze text and determine sentiment scores.
* **Key Insights/Outcome:** 💡 [Summarize findings, e.g., "Analyzed customer reviews, identifying dominant sentiments towards product features and uncovering areas for improvement."]
* **Technologies:** 💻 R (e.g., `tidytext`, `textdata`, `dplyr`, `ggplot2`).
* **Explore Project:** ➡️ [Link to your-repo/r-projects/sentiment-analysis/your_script.Rmd or relevant folder]



💻 Technologies & Libraries
This portfolio primarily leverages the Python ecosystem for data science, including:

Python: Core programming language.
Pandas: Data manipulation and analysis.
NumPy: Numerical computing.
Scikit-learn: Machine learning algorithms and tools.
Matplotlib / Seaborn: Data visualization.
Jupyter Notebooks: Interactive computing environment.
Plotly / Dash: (Optional, if you use them for interactive plots/apps)


▶️ How to Navigate and Run Projects Locally
To explore these projects on your local machine:

1. Clone the repository:
* In Bash:
  - git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
  - cd YOUR_REPOSITORY_NAME

2. Create a virtual environment (recommended):
    python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

3. Install dependencies: Most projects will rely on standard data science libraries. You can try installing common ones or check individual project folders for a `requirements.txt` file.bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Or if a requirements.txt exists:
  pip install -r requirements.txt

4. **Launch Jupyter Notebook:**bash
jupyter notebook
```
5.  Navigate to the respective project folders and open the .ipynb files to view and run the code.

🤝 Connect With Me
I'm always keen to discuss data science, collaborate on projects, or explore new opportunities. Feel free to reach out!

LinkedIn: [Morgan Omodu](https://www.linkedin.com/in/morganomodu/)
Email: omodumorgan@gmail.com
GitHub: [Here](https://github.com/comodu20)

License
This repository is licensed under the MIT License. See the LICENSE file for more details.

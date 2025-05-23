🔬 Data Science Project Portfolio
Welcome to my Data Science Project Portfolio! This repository serves as a showcase of my work in various domains of data science, ranging from predictive modeling and statistical analysis to machine learning algorithms. Each project aims to demonstrate my ability to tackle real-world problems, extract insights from data, and build robust analytical solutions.

🎯 About This Repository
This collection of projects highlights my practical experience with data manipulation, exploratory data analysis (EDA), model building, evaluation, and interpretation. The goal is to provide a clear understanding of my thought process, methodologies, and the impact of the data-driven solutions I develop.

Each project folder typically contains:

Jupyter Notebooks (.ipynb) detailing the analysis.
Relevant datasets (if permissible and small enough).
Markdown files or comments explaining the process and findings.

📂 Project Showcase
Here's an overview of the projects included in this repository. Click on the links to explore the detailed notebooks and code for each.


📈 Predictive Analytics
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



⛓️ Markov Chains
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
* * The accuracy of the Monte Carlo simulation apprach increases as number of steps increases
  * Repeated Matrix Multiplication method is fastest
  * The Left Eigen Vector method (similar to output in singular value decomposition (SVD))

💻 Technologies: Python, numpy, pandas.

➡️ Explore Project: [Here](https://github.com/comodu20/My-Work_ipynb/blob/565ec2332b27923eb3a2fa563deb94e2754b39d7/UEL-DS/Markov%20processes/Markov%20chain%20simulation_1.ipynb)


📉 Linear Regression
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

3. **Install dependencies:** Most projects will rely on standard data science libraries. You can try installing common ones or check individual project folders for a `requirements.txt` file.bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
# Or if a requirements.txt exists:
# pip install -r requirements.txt

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

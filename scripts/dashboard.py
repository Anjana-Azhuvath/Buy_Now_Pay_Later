import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import datetime

st.set_page_config(page_title="BNPL Dashboard", layout="wide")

# Header with Title and Author
st.title("Who Uses Buy Now, Pay Later (BNPL)?")
st.markdown("""
**Author:** Anjana Azhuvath  
**Date:** {}

This dashboard explores trends in BNPL usage using data from the Federal Reserve's 2023 SHED (Survey of Household Economics and Decisionmaking) dataset. 
Use the filters below to explore predicted BNPL usage by age, income, and education, and review key statistical results from a logistic regression model.

**Data Source:** [Federal Reserve Board – SHED Survey](https://www.federalreserve.gov/consumerscommunities/shed.htm)
""".format(datetime.today().strftime('%B %d, %Y')))

# Load Data
df = pd.read_csv("cleaned_shed_data.csv")

# Sidebar Filters
st.sidebar.header("Filter Respondents")
age = st.sidebar.selectbox("Select Age Group", df['age_group'].dropna().unique())
income = st.sidebar.selectbox("Select Income Group", df['income_group'].dropna().unique())
edu = st.sidebar.selectbox("Select Education Level", df['education'].dropna().unique())

# Filtered Subset
filtered = df[(df['age_group'] == age) & (df['income_group'] == income) & (df['education'] == edu)]

# Key Metric: Predicted Probability
st.subheader("Predicted BNPL Usage Probability")
if 'predicted_prob' in filtered.columns and not filtered.empty:
    st.metric("Probability", f"{filtered['predicted_prob'].mean():.2%}")
else:
    st.warning("Predicted probabilities not available in the selected filter.")

# Exploratory Analysis: BNPL Usage by Age, Income, and Education
st.subheader("BNPL Usage by Age, Income, and Education")
fig_ea, ax_ea = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='age_group', y='used_bnpl', hue='income_group', ax=ax_ea)
ax_ea.set_title("BNPL Usage by Age and Income")
plt.legend(title="Income Group")
st.pyplot(fig_ea)

fig_ed, ax_ed = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='education', y='used_bnpl', ax=ax_ed, order=["High School or Less", "Some College", "Bachelor's", "Graduate Degree"])
ax_ed.set_title("BNPL Usage by Education")
plt.xticks(rotation=15)
st.pyplot(fig_ed)

# Regression Model
st.subheader("Logistic Regression Results")
model = smf.logit("used_bnpl ~ C(age_group) + C(income_group) + C(education)", data=df).fit()
st.text(model.summary())

# VIF Table
st.subheader("Variance Inflation Factor (VIF) Analysis")
from patsy import dmatrices
y, X = dmatrices("used_bnpl ~ C(age_group) + C(income_group) + C(education)", data=df, return_type='dataframe')
vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
st.dataframe(vif_data)

# Marginal Effects
st.subheader("Marginal Effects")
mfx = model.get_margeff()
st.text(mfx.summary())

# Full Markdown Block: Interpretation
st.markdown("### Marginal Effects Interpretation: BNPL Usage")
st.markdown("""
This table reports average marginal effects (dy/dx) from a logistic regression predicting BNPL usage based on age, income, and education.
Each value represents the change in probability (in percentage points) of using BNPL compared to a baseline group, holding all other variables constant.

#### Age Effects (Baseline: 18–29)

| Age Group | dy/dx | p-value | Interpretation |
|-----------|-------|---------|----------------|
| 30–44     | +0.68% | 0.473   | Not significantly different from 18–29  |
| 45–59     | –0.85% | 0.384   | Not significantly different  |
| 60+       | –9.21% | < 0.001 | Statistically significantly less likely to use BNPL  |

Older adults (60+) are 9.2 percentage points less likely to use BNPL, controlling for other factors.

#### Income Effects (Baseline: 100K+)

| Income Group | dy/dx | p-value | Interpretation |
|--------------|-------|---------|----------------|
| 25–50K       | +5.95% | < 0.001 | Significantly more likely to use BNPL  |
| 50–100K      | +4.12% | < 0.001 | Also more likely  |
| <25K         | +1.60% | 0.138   | Not significantly different  |

BNPL usage is highest among middle-income groups, especially those earning $25K–50K.

#### Education Effects (Baseline: Bachelor's Degree)

| Education Level       | dy/dx | p-value | Interpretation |
|------------------------|-------|---------|----------------|
| Graduate Degree        | –4.43% | < 0.001 | Statistically significantly less likely to use BNPL  |
| High School or Less    | –1.16% | 0.420   | Not significantly different  |
| Some College           | –3.28% | < 0.001 | Statistically significantly less likely to use BNPL  |

Graduate degree holders are less likely to use BNPL than bachelor's graduates, and those with some college are also less likely.

### Summary Interpretation

BNPL use is highest among middle-income adults with a bachelor's degree, particularly those under 60. Older adults and those with graduate degrees are significantly less likely to use BNPL, indicating differences in financial preferences and fintech adoption across demographics.
""")

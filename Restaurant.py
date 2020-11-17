#Discribtion

'''Historical data of demand for a product-center combination (Weeks: 1 to 145)
Product(Meal) features such as category, sub-category, current price and discount
Information for fulfillment centers like center area, city information, etc.

Source:
https://www.kaggle.com/gauravsahani/food-demand-prediction-dataset'''


#%%
import math # Functions beyond the basic maths
import pandas as pd # For DataFrame and handling
import numpy as np # Array and numerical processing
import seaborn as sns # High level Plotting
from statsmodels.formula.api import ols
import statsmodels.api as sm # Modeling, e.g. ANOVA

#%%
food = pd.read_csv('Food_demand.csv')

#%%
# Examine the data
food.info()
food.shape
food.columns
# %%
#How much is the restaurant income ?
income = sum(food.checkout_price)
# %%
#What is minimum and maximum check out price Weakly?

weeklyChickout = food.groupby('week').checkout_price.sum()

max(weeklyChickout)
min(weeklyChickout)
# %%
#Which dish gives the most income?
mealID = food.groupby('meal_id').checkout_price

range(mealID.checkout_price) 

sns.scatterplot(y='checkout_price', x='meal_id', data=food.groupby('meal_id').sum()['checkout_price'].reset_index())

# %%
#How stable is the weekly chick out price ?
#n of observation in each group
food.groupby('week').count()

sns.histplot(x='checkout_price', data=food.groupby('week').sum()['checkout_price'].reset_index())

#mean
checkoutMean = food.groupby('week')['checkout_price'].mean()

#sd
checkoutSd = food.groupby('week')['checkout_price'].std()

# SEM 
SEM = checkoutSd/ np.sqrt(np.size(food.week))

# 95% CI
CI_lower = checkoutMean - 1.96 * SEM
CI_upper = checkoutMean + 1.96 * SEM

#%%
#Which restaurant center is more popular by the number of orders?
centerID = food.groupby('center_id').num_orders.sum()

sns.scatterplot(y='num_orders', x='center_id', data=food.groupby('center_id').sum()['num_orders'].reset_index())

max(centerID)
min(centerID)

# %%
#How does the centerID influence the number of orders ?
lm_food = ols(" checkout_price ~ center_id", data = food)
results = lm_food.fit()
results.summary()

# compute anova
aov_table = sm.stats.anova_lm(results, typ=2)
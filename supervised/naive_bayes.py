
"""
Naive Bayes from scratch
@Author: Shree Ranga Raju
"""
from __future__ import division
import numpy as np
import pandas as pd

# Create data

# create training data
# create an empty dataframe
data = pd.DataFrame()

# create target variables
data['Gender'] = ['male','male','male','male','female','female','female','female']

#create feature variables
data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
data['Weight'] = [180,190,170,165,100,150,130,150]
data['Foot_size'] = [12,11,12,10,6,8,7,9]

print("Training data is as follows:")
print(data )

#create test data
person = pd.DataFrame()
person['Height'] = [6]
person['Weight'] = [130]
person['Foot_size'] = [8]

print("Test data is shown below:")
print(person)

"""
Bayes Theorem
p(class|data) = p(data|class) * p(class) / p(data)
p(class|data) is called the posterior
p(data|class) is called the likelihood
p(class) is called the prior
p(data) is called the marginal probability

posterior = likelihood * prior / marginal probability

Technically, we only calculate likelihod * prior and ignore marginal probability

p(is_male|data) = p(data|is_male) * p(is_male) / p(data)
p(is_female|data) = p(data|is_female) * p(is_female) / p(data)
We calculate two posteriors as above since there are two classes (male and female)

Gaussian Naive Bayes coz all the features are continuous

posterior(male) = P(male) * p(height|male) * p(weight|male) * p(foot_size|male)
posterior(female) = P(female) * p(height|female) * p(weight|female) * p(foot_size|female)

P(female) is the prior probability. This is just the number of females in the training set
divided by the total # of observations
p(height|female) * p(weight|female) * p(foot_size|female) is the likelihood.

We assume that continous feature values are normally distributed. This means that p(height|female)
is calculated by inputting the required parameters into the probability density function of the
normal distribution

p(height|female) = 1 / (sqrt(2*pi*variance of female height in the data)) * -1 * (e^(obervations height - mean
height of females in the data)^2/2*variance of the female data)

Enough Theory now!
"""

# calculate priors
mf = data['Gender'].value_counts()
# number of males

n_male = mf['male']

#number of females
n_female = mf['female']

# Total rows
n_rows = len(data)

# number of males divided the total number
P_male = n_male / n_rows

P_female = n_female / n_rows

# calculate likelihood

# group the data by gender and calculate mean of the each feature
data_means = data.groupby('Gender').mean()
print(data_means)

# group the data by gender and calculate variance of each feature
data_variance = data.groupby('Gender').var()
print(data_variance)

# Let's seperate out all the variables

# Means for male
male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
male_footsize_mean = data_means['Foot_size'][data_variance.index == 'male'].values[0]

# Variance for male
male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
male_footsize_variance = data_variance['Foot_size'][data_variance.index == 'male'].values[0]

# Means for female
female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
female_footsize_mean = data_means['Foot_size'][data_variance.index == 'female'].values[0]

# Variance for female
female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
female_footsize_variance = data_variance['Foot_size'][data_variance.index == 'female'].values[0]

# Finally, we need to create a function to calculate the probability density of each
# of the terms of the likelihood (eg. p(height|female))
# calculate a function that calculate p(x|y)
def p_x_given_y(x, mean_y, variance_y):

    # input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2) / (2*variance_y))
    return p

# finally it's ready

# numerator of the posterior if the unclassified observation is a male
numerator_post_male = P_male * \
                      p_x_given_y(person['Height'][0], male_height_mean, male_height_variance) * \
                      p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) * \
                      p_x_given_y(person['Foot_size'][0], male_footsize_mean, male_footsize_variance)
print(numerator_post_male)

numerator_post_female = P_female * \
                        p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
                        p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
                        p_x_given_y(person['Foot_size'][0], female_footsize_mean, female_footsize_variance)
print(numerator_post_female)

if numerator_post_male > numerator_post_female:
    print "The new observation is male"
else:
    print "The new observation is female"







# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 00:01:12 2021

@author: SNB
"""

import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

#read Cleveland Heart Disease data
heartDisease = pd.read_csv('heart.csv')
heartDisease = heartDisease.replace('?',np.nan)

#display the data
print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

#Model Bayesian Network
model= BayesianModel([('age','target'),('sex','target'),('exang','target'),
                      ('cp','target'),('target','restecg'),('target','chol')])

#Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)

#computing the Probability of HeartDisease given restecg
print('\n 1. Probability of HeartDisease given evidence= restecg')
q1=HeartDiseasetest_infer.query(variables=['target'],evidence={'restecg':1})
print(q1)

#computing the Probability of HeartDisease given cp
print('\n 2. Probability of HeartDisease given evidence= cp ')
q2=HeartDiseasetest_infer.query(variables=['target'],evidence={'cp':2})
print(q2)

#Part : 2

#Model Bayesian Network
Model = BayesianModel([('age','trestbps'),('age','fbs'), ('sex','trestbps'),
                       ('exang','trestbps'),('trestbps','target'),('fbs','target'),
                       ('target','restecg'),('target','thalach'),('target','chol')])

#Learning CPDs using Maximum Likelihood Estimators
print('\n Learning CPD using Maximum likelihood estimators')
Model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(Model)

#computing the Probability of HeartDisease given Age
print('\n 1. Probability of HeartDisease given Age=30')
q1 = HeartDisease_infer.query(variables=['target'],evidence={'age':28})
print(q1)

#computing the Probability of HeartDisease given cholesterol
print('\n 2. Probability of HeartDisease given cholesterol=100')
q2 = HeartDisease_infer.query(variables=['target'],evidence={'chol':100})
print(q2)

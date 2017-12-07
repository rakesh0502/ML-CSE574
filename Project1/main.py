
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix

sns.set(color_codes=True)

print("UBitName = pbisht2")
print("personNumber = 50247429")

print('UBitName = rakeshsi')
print('personNumber = 50249135')

#Loading the data from the excel sheet

xl = pd.ExcelFile("university data.xlsx")
df = xl.parse("university_data")
df.drop(['rank', 'name', 'Grad Student No.', 'TT Faculty', 'Lecturers', 'G-TT Ratio', 'G-TTL Ratio'], axis=1, inplace=True)

df_na = df.dropna()
df_cs_score_research = df_na.drop(['Admin Base Pay$', 'Tuition(out-state)$'], axis=1)
df_research_admin_base_pay = df_na.drop(['CS Score (USNews)', 'Tuition(out-state)$'], axis=1)
df_admin_base_pay_cs_score = df_na.drop(['Research Overhead %', 'Tuition(out-state)$'], axis=1)
df_cs_score_tuition = df_na.drop(['Research Overhead %', 'Admin Base Pay$'], axis=1)
df_research_tuition = df_na.drop(['CS Score (USNews)', 'Admin Base Pay$'], axis=1)
df_admin_base_tuition = df_na.drop(['CS Score (USNews)', 'Research Overhead %'], axis=1)
data = df_na.as_matrix()

mu = np.mean(data, axis=0)
var = np.var(data, axis=0)
sigma = np.std(data, axis=0)

print('mu1 =', round(mu[0], 3))
print('mu2 =', round(mu[1], 3))
print('mu3 =', round(mu[2], 3))
print('mu4 =', round(mu[3], 3))

print('var1 =', round(var[0], 3))
print('var2 =', round(var[1], 3))
print('var3 =', round(var[2], 3))
print('var4 =', round(var[3], 3))

print('sigma1 =', round(sigma[0], 3))
print('sigma2 =', round(sigma[1], 3))
print('sigma3 =', round(sigma[2], 3))
print('sigma4 =', round(sigma[3], 3))

covarianceMat = np.cov(data, rowvar=False)
for i in range(0, len(covarianceMat)):
    for j in range(0, len(covarianceMat[i])):
        covarianceMat[i][j] = round(covarianceMat[i][j], 3)
print('covarianceMat =\n', covarianceMat)

correlationMat = np.corrcoef(data, rowvar=False)
for i in range(0, len(correlationMat)):
    for j in range(0, len(correlationMat[i])):
        correlationMat[i][j] = round(correlationMat[i][j], 3)
print('correlationMat =\n', correlationMat)

row_size = len(data)
col_size = len(data[0])
data_trans = data.transpose()
pdf_uni = np.empty(shape=(col_size, row_size))
for i in range(0, col_size):
    for j in range(0, row_size):
        pdf_uni[i][j] = (math.exp(-0.5 * ((math.pow((data_trans[i][j] - mu[i]), 2)) / var[i]))) \
                        / (math.pow((2 * math.pi * var[i]), 0.5))

likelihood_arr = np.empty(shape=row_size)
logLikelihood = 0

for col in range(0, row_size):
    likelihood_arr[col] = (pdf_uni[0][col]) * (pdf_uni[1][col]) * (pdf_uni[2][col]) * (pdf_uni[3][col])
    likelihood_arr[col] = math.log(likelihood_arr[col])
    logLikelihood = logLikelihood + likelihood_arr[col]

print("logLikelihood = %0.3f" % logLikelihood)

logLikelihood_multi = 0
cov_arr_inv = np.linalg.inv(covarianceMat)
cov_arr_det = np.linalg.det(covarianceMat)
pdf_multi = np.empty(shape=row_size)
vector = np.empty(shape=(row_size, col_size))


for i in range(0, row_size):
    vector = np.array([np.subtract(data[i], mu)])
    pdf_multi[i] = ((math.exp(-0.5 * (vector.dot(cov_arr_inv.dot(np.transpose(vector)))))) \
                    / ((math.pow(2 * math.pi, 2)) * (math.pow(cov_arr_det, 0.5))))
    logLikelihood_multi = logLikelihood_multi + math.log(pdf_multi[i])

print("logLikelihood MultiVariate = %0.3f" % logLikelihood_multi)

sns.pairplot(df_cs_score_research, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_cs_score_research_plot.jpg")
plt.show()

sns.pairplot(df_research_admin_base_pay, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_research_admin_base_pay_plot.jpg")
plt.show()

sns.pairplot(df_admin_base_pay_cs_score, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_admin_base_pay_cs_score_plot.jpg")
plt.show()

sns.pairplot(df_cs_score_tuition, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_cs_score_tuition_plot.jpg")
plt.show()

sns.pairplot(df_research_tuition, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_research_tuition_plot.jpg")
plt.show()

sns.pairplot(df_admin_base_tuition, size=2.5, aspect=2.5, kind='reg')
plt.savefig("pairwise_admin_base_tuition_plot.jpg")
plt.show()



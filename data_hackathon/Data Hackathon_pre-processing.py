#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot

train = pd.read_csv('Train_nyOWmfK.csv', encoding = 'iso-8859-1')
test = pd.read_csv('Test_bCtAN1w.csv', encoding = 'iso-8859-1')

#Combine into data
train['source'] = 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)

#Check for missing values
data.apply(lambda x : sum(x.isnull()))

#Look at categories of all object variables
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print ('\nFrequency count for variable %s'%v)
    print (data[v].value_counts())

#Handle Individual Variables
#City Variable
len(data['City'].unique())
#drop city because too many unique
data.drop('City',axis=1,inplace=True)

#Determine Age from DOB
#Create age variable:
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))
data['Age'].head()

#drop DOB:
data.drop('DOB',axis=1,inplace=True)

#EMI_Load_Submitted
data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')

#Majority values missing so I'll create a new variable stating whether this is missing or note:
data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)

#drop original vaiables:
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)

#Employer Name
len(data['Employer_Name'].value_counts())

#I'll drop the variable because too many unique values. Another option could be to categorize them manually
data.drop('Employer_Name',axis=1,inplace=True)

#Existing EMI
data.boxplot(column='Existing_EMI',return_type='axes')

data['Existing_EMI'].describe()

#Impute by median (0) because just 111 missing:
data['Existing_EMI'].fillna(0, inplace=True)

#Interest Rate
#Majority values missing so I'll create a new variable stating whether this is missing or note:
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
print (data[['Interest_Rate','Interest_Rate_Missing']].head(10))

data.drop('Interest_Rate',axis=1,inplace=True)

#Lead Creation Date
#Drop this variable because doesn't appear to affect much intuitively
data.drop('Lead_Creation_Date',axis=1,inplace=True)

#Loan Amount and Tenure applied
#Impute with median because only 111 missing:
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)

#Loan Amount and Tenure selected
#High proportion missing so create a new var whether present or not
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)

#Remove old vars
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)

#Remove logged-in
data.drop('LoggedIn',axis=1,inplace=True)

#Remove salary account
#Salary account has mnay banks which have to be manually grouped
data.drop('Salary_Account',axis=1,inplace=True)

#Processing_Fee
#High proportion missing so create a new var whether present or not
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
#drop old
data.drop('Processing_Fee',axis=1,inplace=True)

#Source
data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()

#Final Data
data.apply(lambda x: sum(x.isnull()))


#Numerical Coding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])

#One-Hot Coding
data = pd.get_dummies(data, columns=var_to_encode)
data.columns


#Separate train & test

train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)

train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)
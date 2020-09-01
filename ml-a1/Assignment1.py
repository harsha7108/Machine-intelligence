'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
    entropy=0
    target=df.iloc[:,-1]
    n=len(target) 
    uniq, count = np.unique(target,return_counts=True)
    for i in range(len(uniq)):
        entropy-=(count[i]/n * np.log2(count[i]/n))
    return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
    in_avg=[]
    count=df.iloc[:,-1].value_counts()
    total=sum(count)
    target=df[attribute]
    attr_type=np.unique(target)
    for i in attr_type:
        entropy=0
        p=df.loc[(df[attribute]==i)]
        frequency=p.iloc[:,-1].value_counts(normalize=True)
        v=p.iloc[:,-1].value_counts()
        values=sum(v)
        for i in frequency:
            entropy-=i*(np.log2(i))
        information=(values/total)*entropy
        in_avg.append(information)  
    entropy_of_attribute = sum(in_avg)
    return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
    entropy=get_entropy_of_dataset(df)
    information=get_entropy_of_attribute(df,attribute)
    information_gain = entropy-information
    return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
    information_gains={}
    k=len(df.columns)
    j=0
    for i in df.columns:
        if(j!=k-1):
            ig=get_information_gain(df,i)
            information_gains[i]=ig
        j+=1
    selected_column=max(information_gains,key=information_gains.get)      
        
    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

    return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''

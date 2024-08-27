# -*- coding: utf-8 -*-
"""
Created on Sat Sept 23 17:53:24 2022

@author: Firo Obeid
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
# import matplotlib.pyplot as plt
import sys
# import seaborn as sns 

pd.set_option('max_colwidth', 800)


'''
Built a main function that calculates, NPV, IRR, Profitability index,
payback period and displays pandas of dicounted cash flows(npv_data) 
and undiscounted cash flows(dfcash)
'''

def discounting(r, n):
    '''
    This function will be called from my mother function to discount the cash flows
    '''
    discount_factor = [] #The discounting factors will be stored in a list
    for h in range(0, n + 1): # loop to get discount factors and append to a list
        rate = 1 / ((1 + r) ** h)
        discount_factor.append(rate)
    return discount_factor

def calc_k(CFS, intial_cost):
    '''
    Calculate k that gives back last period of Cashflows not negative vs intial_cost else = 1
    '''
    global k
    trials_till_positive = intial_cost
    if (sum(CFS) + trials_till_positive) > 0:
        i = 0
        while trials_till_positive < 0:
            trials_till_positive += CFS[i]
            i+=1
        k = i - 1   
    else:
        k=1

# Main Function for the whole code:
def calc_npv(n, intial_cost, cash_flows, r): 

    """
    A full mother function the calculate NPV, IRR and display pandas. 
    All of that is done by passing the cashflows, period and rate of return(when needed)

    Parameters
    ----------
    n : nt or float
        Term Maturity
    intial_cost :
        Downpayment (should be negative since outflow)
    the 2nd param
    cash_flows : array-like,
    r : float


    Returns
    -------
    string
    a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """
    from watermark import watermark
    print(watermark())
    print(watermark(iversions=True, globals_=globals()))
    r/=100
    npv_data = pd.DataFrame({'Cash_Flows': [i for i in range(0,n + 1)]}, index = [i for i in range(0, n + 1)]) #the iterations will take whatever cash flows assigned when my function is called in separate problems
    npv_data.index.name = 'Year' #The index is set to be named "Years'. The number of years is passed through function call
    npv_data.loc[0]['Cash_Flows'] = intial_cost #index zero in colmun with intial cost. The zero index was empty and reserved for the intial cost as stated
    npv_data.Cash_Flows.loc[[i for i in range(1, n + 1)]] = cash_flows #tried several formats until this worked to change all cash flows through passing a list to the function and updating the iterable cash flow series. Note that range starts at 1 not 0 to keep 0 for intial cost reserved
    npv_data['Discounting factors'] = pd.Series(discounting(r,n)) #panda columns are eventually series so I added a new series of the discounted cash flows through an outter function coded earlier
    npv_data['Discounted_Cash_Flows'] = npv_data.apply(np.prod, axis = 1) #apply method used to multiply cash flows column with discounting factors column to give new discounted cash flows column. Axis = 1 resembles columns and 0 for rows
    df_cash_flow_only = pd.DataFrame({'Cash_Flows': [i for i in range(0,n + 1)]}, index = [i for i in range(0, n + 1)]) #dataframe for non discounted cash flows for payvack period cacl. and displaying purposes
    df_cash_flow_only.index.name = 'Year'
    df_cash_flow_only.loc[0]['Cash_Flows'] = intial_cost
    df_cash_flow_only.Cash_Flows.loc[[i for i in range(1, n + 1)]] = cash_flows
    npv = npf.npv(r, npv_data.loc[:]['Cash_Flows']) #Pass cash flow column from dataframe. I could have summed up the discounted cash flows + intial cost, but kept a general frmat for calculation purposes
    irr = round((npf.irr(npv_data.loc[:]['Cash_Flows']) * 100), 4)
    prof_index = 0 # I was getting a runtime warning, thus I adjusted that be add if_st so that if one of the problems have intial_cost = 0 dont excute
    
    if intial_cost != 0:
        prof_index = round((npf.npv(r, npv_data.loc[:]['Cash_Flows']) + (-intial_cost)) / (-intial_cost), 3) # to caclculate PI through removing the intial cost from discounted cash flows then dividind the PV of DCF / initial cost

    if r == 0: #if rate is zero the cash flows oayback will be calculated and passed by providing the "k" iterable in the call function, which is the last year that cumulative cash flows are <=0.
        calc_k(cash_flows, intial_cost)
        residual_CF = intial_cost + sum(cash_flows[:k])
        left_over_CF = -residual_CF / cash_flows[k]
        period = round((k  + left_over_CF), 4)  
    else: #if r is not zero, we calculate the discounted cash flows payback. I appended to a series the discounting factors through calling th outside function and turning that series to a list. The index 0 is deleted since its is 1 and we want the first element in D_C_F to discount first element in cash_flows list
        D_C_F =  pd.Series(discounting(r,n)).tolist()
        del D_C_F[0]
        discounted_list =  [x * D_C_F[i] for i, x in enumerate(cash_flows)] #for every index in cash_flow list, the respective index in D_C_F will be multiplied to get back the discounted cash flows. I used this method instead of retriving from pandas since accessing pandas in such a way don't give back reuired elements.
        calc_k(discounted_list, intial_cost)
        residual_CF = intial_cost + sum(discounted_list[:k])
        left_over_CF = -residual_CF /discounted_list[k] #fractional payback that goes on to the kth iteration 
        period = round((k  + left_over_CF), 4)
    print(pd.DataFrame({"npv":npv,"irr":irr,"period":period,"prof_index":prof_index}, index = [0]))
    print(npv_data)
    return npv, npv_data, irr, df_cash_flow_only, period, prof_index # when calling the function I have to call all returned variables and fill all parameters. Some parameters are passed to allows the function to return a value without them yielding a significant implication on the functions calculations


npv, npv_data, irr, dfcash, period, PI = calc_npv(20, -3375000 , [3850000] * 20, 9.5754)

import pandas as pd 
import numpy as np
import pdb
import statsmodels.api as sm
from sklearn import linear_model

class ukdata(object):
    def __init__(self):

        self.dfacc = pd.read_csv('./Accidents0514.csv')
        self.dfcas = pd.read_csv('./Casualties0514.csv')
        self.dfveh = pd.read_csv('./Vehicles0514.csv')
        
    
    def qn1(self):
        urban_acc = (self.dfacc['Urban_or_Rural_Area']==1).sum()
        total_acc = self.dfacc['Urban_or_Rural_Area'].count()
        frac_urban_acc = float(urban_acc)/float(total_acc)
        print frac_urban_acc
  

    def qn2(self):

        df_accidents = pd.read_csv('Accidents0514.csv')
        count0 = 0
        count1 = 0
        count2 = 0
        count3= 0
        count4 = 0
        count5= 0
        count6 = 0
        count7 = 0
        count8 = 0
        count9 =0
        null_sum = pd.isnull(df_accidents['Date']).sum()
        #print("null_sum", null_sum)
        total_sum = df_accidents['Date'].count()
        #print(df_accidents['Date'])
        actual_sum=total_sum - null_sum
        #print("total_sum", total_sum)

         
        for i in range((total_sum - null_sum)):
            k = df_accidents['Date'][i].split("/")
            if int(k[2]) == 2005:
                count0 =count0 +1
            
            if int(k[2]) == 2006:
                count1 = count1 +1
            if int(k[2]) == 2007:
                count2 = count2 + 1
            if int(k[2]) == 2008:
                count3 = count3 + 1
            if int(k[2]) == 2009:
                count4 =count4 +1
            if int(k[2]) == 2010:
                count5 = count5 +1
            if int(k[2]) == 2011:
                count6 = count6 + 1
            if int(k[2]) == 2012:
                count7 = count7 + 1
            if int(k[2]) == 2013:
                count8 = count8 + 1
            if int(k[2]) == 2014:
                count9 = count9 + 1
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0])
        
        y = np.array([float(count0), float(count1), float(count2), float(count3), float(count4), float(count5), float(count6), float(count7), float(count8), float(count9)])
        
        regression = linear_model.LinearRegression()
        x = sm.add_constant(x)
        value = sm.OLS(y, x).fit()
        print value.params
        print value.summary()


    def qn3(self):
        df_accidents = pd.read_csv('Accidents0514.csv')
        df_vehicles = pd.read_csv('Vehicles0514.csv')
        count = 0
        null_sum =0

        #null_sum = pd.isnull(df_accidents['Urban_or_Rural_Area']).sum()
        list_of_accidents1 =df_accidents.query('Weather_Conditions == 2 | Weather_Conditions == 3 | Weather_Conditions == 5 | Weather_Conditions == 6')['Accident_Index']

        print(len(list_of_accidents1))
        
        list_of_accidents2 =df_vehicles.query('Skidding_and_Overturning != 0 | Skidding_and_Overturning != -1')['Accident_Index']
        print(len(list_of_accidents2))

        #list_of_accidents3 =df_accidents[df_accidents['Weather_Conditions'] == 1]['Accident_Index']
        list_of_accidents3 =df_accidents.query('Weather_Conditions == 1')['Accident_Index']
        print(len(list_of_accidents3))


        print(set(list_of_accidents1).issubset(list_of_accidents2))
        ans1 = len(set(list_of_accidents2).intersection(list_of_accidents1))

        ans2 = len(set(list_of_accidents2).intersection(list_of_accidents3))
        print(ans1)
        print(ans2)
        print(float(ans1)/ans2)




    def qn4(self):
        
        police_area = []
        for itr in set(self.dfacc['Police_Force']):
            idx_police = (self.dfacc['Police_Force']==itr)
            police_longstd = self.dfacc['Longitude'][idx_police].std()
            police_latstd = self.dfacc['Latitude'][idx_police].std()
            police_area.append(np.pi*police_longstd*police_latstd)
    
        ans4 = max(police_area)
        
        print 'area in degrees ={}'.format(ans4)
        ans4k = max(police_area)*111*111
        print 'area in km = {}'.format(ans4k)


    def qn5(self):
        
        self.dfacc['hour'] = pd.to_datetime(self.dfacc['Time'], format = '%H:%M').dt.hour
       
        cur_normacc = []
        for itr in range(24):
            acc_curhour = self.dfacc['Accident_Severity'][self.dfacc['hour']==itr]
            cur_fatalacc = (acc_curhour==1).sum()
            cur_acc = acc_curhour.count()
            cur_normacc.append(float(cur_fatalacc)/float(cur_acc))
        dang_hour = np.argmax(np.array(cur_normacc))
        #grouped_acseverty = self.dfacc['Accident_Severity'].groupby(self.dfacc['hour'])
        #categories = pd.cut(self.dfacc['Time'], bins)
        print 'accident radio={}'.format(cur_normacc[4])




    def qn7(self):
        

        df_accidents = pd.read_csv('Accidents0514.csv')
        df_vehicles = pd.read_csv('Vehicles0514.csv')

        male_accidents = df_vehicles.query('Sex_of_Driver == 1')['Accident_Index']
        female_accidents = df_vehicles.query('Sex_of_Driver == 2')['Accident_Index']

        fatal_accidents = df_accidents.query('Accident_Severity == 1')['Accident_Index']
        
        ans1 = len(set(fatal_accidents).intersection(female_accidents))
        ans2 = len(set(fatal_accidents).intersection(male_accidents))
        
        print(float(ans2)/ans1)

    def qn6(self):
        
        df_accidents = pd.read_csv('Accidents0514.csv')
        unique_speed_limit = df_accidents.Speed_limit.unique()
        ratio = []
        for i in unique_speed_limit:
            print("speed_limit", i)
            no_of_accidents = df_accidents.query('Speed_limit == "%s"' %i)['Accident_Index']
            count_of_accidents = (len(no_of_accidents))
            print("No_Of_Accidents", count_of_accidents)
            no_of_casualities = df_accidents.query('Speed_limit == "%s"'%i)['Number_of_Casualties']
            count_of_deaths = sum(no_of_casualities)
            print("Number_Of_Deaths", count_of_deaths)
            ans = float(count_of_accidents)/count_of_deaths
            ratio.append(ans)
            print("ratio", ratio)
        print("answer", np.corrcoef(unique_speed_limit, ratio))
        
    def qn8(self):

        df_vehicles = pd.read_csv('Vehicles0514.csv')
        legal_drivers = df_vehicles.query('Age_of_Driver >= 17')['Age_of_Driver'].unique()
        print legal_drivers
        logx = []
        for i in legal_drivers:
            no_of_accidents= df_vehicles.query('Age_of_Driver == "%s"' %i)['Accident_Index']
            print("age_of_legal_driver", i)
            count_of_accidents = no_of_accidents.count()
            print("No_Of_Accidents", count_of_accidents)
            logx.append(count_of_accidents)

        legal_drivers = sm.add_constant(legal_drivers)
        value = sm.OLS(logx, legal_drivers).fit()
        print value.params
        print value.summary()


if __name__=='__main__':
    o = ukdata()
    o.qn1()
    o.qn2()
    o.qn4()
    o.qn5()
    o.qn6()
    o.qn7()
    o.qn8()

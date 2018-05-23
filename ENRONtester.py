
import os
import pickle
# ### Import the file which contain the dataset to our variable
# Load the dictionary containing the dataset
with open(os.getcwd()+"/final_project_dataset.pkl", "rb") as data_file:
    enron_data = pickle.load(data_file)
# ### Converting the dataset from a python dictionary to a pandas dataframe
enron_data.keys()
#print (enron_data.keys())
#print (len (enron_data.keys()))


# value of stock of pois


print (enron_data['LAY KENNETH L']['from_this_person_to_poi'])
print (enron_data['LAY KENNETH L']['from_messages'])
print (enron_data['LAY KENNETH L']['from_poi_to_this_person'])

print (len(enron_data['SKILLING JEFFREY K']))

print (len(enron_data))
count = 0
for user in enron_data:
    if enron_data[user]['poi'] == True:
        count+=1
print (count)
def poiEmails():
    email_list = ["kenneth_lay@enron.net",
            "kenneth_lay@enron.com",
            "klay.enron@enron.com",
            "kenneth.lay@enron.com",
            "klay@enron.com",
            "layk@enron.com",
            "chairman.ken@enron.com",
            "jeffreyskilling@yahoo.com",
            "jeff_skilling@enron.com",
            "jskilling@enron.com",
            "effrey.skilling@enron.com",
            "skilling@enron.com",
            "jeffrey.k.skilling@enron.com",
            "jeff.skilling@enron.com",
            "kevin_a_howard.enronxgate.enron@enron.net",
            "kevin.howard@enron.com",
            "kevin.howard@enron.net",
            "kevin.howard@gcm.com",
            "michael.krautz@enron.com"
            "scott.yeager@enron.com",
            "syeager@fyi-net.com",
            "scott_yeager@enron.net",
            "syeager@flash.net",
            "joe'.'hirko@enron.com",
            "joe.hirko@enron.com",
            "rex.shelby@enron.com",
            "rex.shelby@enron.nt",
            "rex_shelby@enron.net",
            "jbrown@enron.com",
            "james.brown@enron.com",
            "rick.causey@enron.com",
            "richard.causey@enron.com",
            "rcausey@enron.com",
            "calger@enron.com",
            "chris.calger@enron.com",
            "christopher.calger@enron.com",
            "ccalger@enron.com",
            "tim_despain.enronxgate.enron@enron.net",
            "tim.despain@enron.com",
            "kevin_hannon@enron.com",
            "kevin'.'hannon@enron.com",
            "kevin_hannon@enron.net",
            "kevin.hannon@enron.com",
            "mkoenig@enron.com",
            "mark.koenig@enron.com",
            "m..forney@enron.com",
            "ken'.'rice@enron.com",
            "ken.rice@enron.com",
            "ken_rice@enron.com",
            "ken_rice@enron.net",
            "paula.rieker@enron.com",
            "prieker@enron.com",
            "andrew.fastow@enron.com",
            "lfastow@pdq.net",
            "andrew.s.fastow@enron.com",
            "lfastow@pop.pdq.net",
            "andy.fastow@enron.com",
            "david.w.delainey@enron.com",
            "delainey.dave@enron.com",
            "'delainey@enron.com",
            "david.delainey@enron.com",
            "'david.delainey'@enron.com",
            "dave.delainey@enron.com",
            "delainey'.'david@enron.com",
            "ben.glisan@enron.com",
            "bglisan@enron.com",
            "ben_f_glisan@enron.com",
            "ben'.'glisan@enron.com",
            "jeff.richter@enron.com",
            "jrichter@nwlink.com",
            "lawrencelawyer@aol.com",
            "lawyer'.'larry@enron.com",
            "larry_lawyer@enron.com",
            "llawyer@enron.com",
            "larry.lawyer@enron.com",
            "lawrence.lawyer@enron.com",
            "tbelden@enron.com",
            "tim.belden@enron.com",
            "tim_belden@pgn.com",
            "tbelden@ect.enron.com",
            "michael.kopper@enron.com",
            "dave.duncan@enron.com",
            "dave.duncan@cipco.org",
            "duncan.dave@enron.com",
            "ray.bowen@enron.com",
            "raymond.bowen@enron.com",
            "'bowen@enron.com",
            "wes.colwell@enron.com",
            "dan.boyle@enron.com",
            "cloehr@enron.com",
            "chris.loehr@enron.com"
        ]
    return email_list
#POIEmail
print (len(poiEmails()))

enron_data.keys()
enron_data['SKILLING JEFFREY K'].keys()
#Skillings Jeffrey
print (enron_data['SKILLING JEFFREY K'].keys())
print (enron_data['PRENTICE JAMES']['total_stock_value'])
print (enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print (enron_data['SKILLING JEFFREY K']['exercised_stock_options'])
print (sorted(enron_data.keys()))
print (enron_data['SKILLING JEFFREY K']['total_payments'])
print(enron_data['FASTOW ANDREW S']['total_payments'])

print (enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

count_salary = 0
count_email = 0
for key in enron_data.keys():
    if enron_data[key]['salary'] != 'NaN':
        count_salary+=1
    if enron_data[key]['email_address'] != 'NaN':
        count_email+=1
print (count_salary)
print (count_email)

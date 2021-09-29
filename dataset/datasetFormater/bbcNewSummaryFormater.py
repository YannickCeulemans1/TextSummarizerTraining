import pandas as pd

data = []
# Voor het berekenen van de gemiddelde artikel lengte 
aLen = []
# Voor het berekenen van de gemiddelde sammenvatting lengte 
sLen = []

def getData(subject, amount):
    global data
    global aLen
    global sLen
    
    for i in range(amount):  
        dataItem = [] 
        dataItem.append(subject)

        # Maak file naam met nummer met nul(en)voor als kleiner als 3 digets
        file = f"{(i+1):03d}.txt"  

        # Voeg articel text toe aan lijst
        articel = open('datasets/datasetFormater/bbcNewsSummary/News_Articles/'+subject+'/'+file, 'r').read()
        dataItem.append(articel)
        aLen.append(len(articel))

        # Voeg summarie text toe aan lijst
        summary = open('datasets/datasetFormater/bbcNewsSummary/Summaries/'+subject+'/'+file, 'r').read()
        dataItem.append(summary) 
        sLen.append(len(summary))

        data.append(dataItem)
    

getData("business", 510)
getData("entertainment", 386)
getData("politics", 417)
getData("sport", 511)
getData("tech", 401)

print("article average:",(sum(aLen))/len(aLen))
print("summarie average:",(sum(sLen))/len(sLen))

df = pd.DataFrame(data,columns =['Subject','Articles', 'Summaries'])

df.to_csv("datasets/BBCnews.csv", index = False)


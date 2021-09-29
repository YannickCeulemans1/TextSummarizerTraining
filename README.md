## Data set
De dataset is een BBC news dataset van meer dan 2000 artikels met elks een samenvatting.  
Deze is geformateerd naar een simpel csv bestand dat makkelijk te gebruiken is voor de summarizer.

## Data wrangling
Er zijn 4 stappen om de data van de dataset om te vormen naar de juiste data voor het trainen van het logistic regression model. 

1_dataPreProcessing = Hier worden de embeddings van de artikels en samenvattingen gegenereerd door BERT, dit kan enkele uren duren.  
2_labelCalculation = Hier worden de labels voor deze artikels-samenvatting pairs gegenereerd.  
3_dataFormatting = De data is nog niet in een juist formaat voor het logistic regression model om te gebruiken dus wordt het hier geformateerd.  
4_train_test_split = De data wordt hier opgesplitst in train data om het model met te trainen en test data om achteraf het resultaat van het model met te testen. 

## Data
Tussen elke stap wordt de data opgeslagen in een pickle bestand, in de map "data". Ook het getraind model en de test resultaten hiervan worden in die map opgeslagen.  

## Logistic regression model
In het "logisticRegressionModel" file wordt het model getrained en getest. 

## Jupyter notebook
Elke script heeft zowel het orginele python script als een nieuwer jupyter notebook bestand als documentatie.
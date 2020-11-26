import csv, json

csvFilePath = 'metrics.csv'
jsonFilePath = 'metrics.json'

data = {}

with open(csvFilePath) as csvFile:
    csvReader = csv.DictReader(csvFile)
    for rows in csvReader:
        id = rows['epoch']
        data[id] = rows

with open(jsonFilePath, 'w+') as jsonFile:
    jsonFile.write(json.dumps(data, indent=4))


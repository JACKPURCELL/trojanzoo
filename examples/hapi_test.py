import hapi
hapi.config.data_dir = "/home/jkl6486/HAPI" 
hapi.download()
df = hapi.summary()
print(df)
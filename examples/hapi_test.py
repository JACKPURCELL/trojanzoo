import hapi
hapi.config.data_dir = "/home/jkl6486/HAPI" 
# hapi.download()
# df = hapi.summary()
# print(df)
# def get_predictions(
#     task: Union[str, List[str]] = None,
#     dataset: Union[str, List[str]] = None,
#     api: Union[str, List[str]] = None,
#     date: Union[str, List[str]] = None,
#     include_dataset: bool = None,
# )

predictions =  hapi.get_predictions(task="fer", dataset="rafdb", date="22-05-23", api=["google_fer"])
print(predictions)
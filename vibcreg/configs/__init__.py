import yaml

stream = open("example_ucr.yaml", 'r')
config = yaml.load(stream, Loader=yaml.FullLoader)

print(config)
print(config["ucr_dataset_name"]["value"])

print(dict(yaml=config))
from moa_classifier_train import Config

print(f"Config.IMG_SIZE = {Config.IMG_SIZE}")
print(f"Type: {type(Config.IMG_SIZE)}")

if isinstance(Config.IMG_SIZE, (list, tuple)):
    print(f"Is tuple/list with values: {Config.IMG_SIZE}")
else:
    print(f"Is scalar with value: {Config.IMG_SIZE}")

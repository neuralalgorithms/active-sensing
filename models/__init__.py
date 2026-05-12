# previously: mapped model class -> string, which was rather silly. 
# if must use strings, better way would be a registry with decorators so you don't have to update the list manually
# this is a compromise
from .cnn import (
    CNN,  # noqa: F401
    SimpleCNN, # noqa: F401
    MaskedCNN, # noqa: F401
    SmallCNN, # noqa: F401
    SuperSmallCNN, # noqa: F401
    ModeratelySmallCNN # noqa: F401
)

# previous code
"""
def get_model(name, **kwargs):
    models = {
        "cnn": cnn.CNN,
        "simplecnn": cnn.SimpleCNN,
        "maskedcnn": cnn.MaskedCNN,
        "smallcnn": cnn.SmallCNN,
        "supersmallcnn": cnn.SuperSmallCNN,
        "moderatelysmallcnn": cnn.ModeratelySmallCNN,
        "mlp": MLP,
    }

    if name not in models:
        raise ValueError(f"Model {name} not found. Available: {list(models.keys())}")

    return models[name](**kwargs)
"""
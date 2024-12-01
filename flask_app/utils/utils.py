import json
import numpy as np


def process_predictions(predictions, classes_dictionary_json):
    import numpy as np
    import json

    with open(classes_dictionary_json, 'r') as file:
        class_dictionary = json.load(file)

    # Sort classes by keys
    classes = [class_dictionary[key] for key in sorted(class_dictionary.keys())]
    classes = np.array(classes)
    
    # Ensure predictions is an array
    if np.isscalar(predictions):  # If it's a single value, wrap it in a numpy array
        predictions = np.array([predictions])
    
    # Sort predictions and get top indices
    top_predictions_indices = np.argsort(predictions)[::-1]
    
    return predictions[top_predictions_indices], classes[top_predictions_indices]



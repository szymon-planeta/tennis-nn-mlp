import pickle

def serialize(data, output_file):
    with open(output_file, "wb") as out:
        pickle.dump(data, out)
        
def deserialize(input_file):
    with open(input_file, "rb") as in_file:
        data = pickle.load(in_file)
    return data
import json

def save_namespace(FLAGS_dict, out_path):
    with open(out_path, 'w', encoding='utf8') as foutput:
        json.dump(FLAGS_dict, foutput)
        
def load_namespace(input_path):
    with open(input_path, 'r', encoding='utf8') as finput:
        FLAGS_dict = json.load(finput)
    return FLAGS_dict

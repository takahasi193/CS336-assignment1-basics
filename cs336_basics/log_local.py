import json
def log_save_to_disk(output:str,log:dict[str,int]):
    with open(output,"a",encoding="utf-8") as f:
        f.write(json.dumps(log)+'\n')
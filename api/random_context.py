import json
import random
import time

def get_random_context(dataset_name):
    """
    스무고개 질문지가 담긴 text_dict.json에서 임의의 context를 선정해 context, context id를 반환
    """
    random.seed(time.time())
    print(f'dataset_name={dataset_name}')
    with open(dataset_name, 'r') as file:
        json_data = json.load(file)

    categories = list(json_data.keys())
    rnd_category = random.choice(categories)
    print(f'category={categories}, rnd_category={rnd_category}')
    numbers = list(json_data[rnd_category].keys())
    rnd_number = random.choice(numbers)
    print(f'numbers={numbers}, rnd_number={rnd_number}')
    raw_datasets = json_data[rnd_category][rnd_number]

    return list(raw_datasets.items())[0][1], numbers
    
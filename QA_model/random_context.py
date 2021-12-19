import json, random, time, os

def get_random_context_from_category(dataset_name: str, category: str):
    """
    dataset_anem: 스무고개 질문지가 담긴 directory 이름
    category: 스무고개 카테고리 이름
    """
    assert os.path.isdir(dataset_name)
    path = os.path.join(dataset_name, category)
    assert os.path.isdir(path)

    candidates = []
    with os.scandir(path) as dir_list:
        for entry in dir_list:
            if entry.name.endswith('txt') and entry.is_file():
                candidates.append(entry)

    with open(random.choice(candidates).path, 'r', encoding='utf-8') as file:
        contexts = file.readlines()
    return ' '.join(contexts)

def get_random_context_text_json(dataset_name):
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
    
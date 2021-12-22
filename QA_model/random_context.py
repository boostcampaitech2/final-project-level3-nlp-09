import json, random, time, os

def get_context(dataset_path: str, category: str, context_name: str):
    """
    dataset_path: 스무고개 질문지가 담긴 directory 이름
    return: 
        category, context, answer
    """
    assert os.path.isdir(dataset_path)
    assert os.path.isdir(os.path.join(dataset_path, category))
    assert os.path.exists(os.path.join(dataset_path, category, context_name + '.txt'))

    target_path = os.path.join(dataset_path, category, context_name + '.txt')
    with open(target_path, 'r', encoding='utf-8') as file:
        contexts = file.readlines()

    context = ' '.join(contexts)
    return context

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
    
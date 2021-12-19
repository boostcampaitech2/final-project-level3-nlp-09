import json, random, time, os

def get_random_context(dataset_path: str):
    """
    dataset_path: 스무고개 질문지가 담긴 directory 이름
    return: 
        category, context, answer
    """
    assert os.path.isdir(dataset_path)
    random.seed(time.time())

    category_candidates= []

    # Append candidates of category
    with os.scandir(dataset_path) as entries:
        for entry in entries:
            if entry.is_dir():
                category_candidates.append(entry)

    # Select category
    path = random.choice(category_candidates)
    category = path.name

    candidates = []
    # Append candidates of contexts
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.name.endswith('txt') and entry.is_file():
                candidates.append(entry)

    # Select context!
    target = random.choice(candidates)
    with open(target.path, 'r', encoding='utf-8') as file:
        contexts = file.readlines()

    context = ' '.join(contexts)
    answer = target.name[:-4]
    return category, context, answer

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
    
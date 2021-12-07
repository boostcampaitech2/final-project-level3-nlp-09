import wikipediaapi

wiki = wikipediaapi.Wikipedia('ko')
while(True):
    e_flag = input("종료하려면 2 입력: ")
    if e_flag == '2':
        break
    words = input("원하는 단어 입력: ")
    page_py = wiki.page(words)
    if page_py.exists() == True:
        #print(page_py.text)
        f_name = words+'.txt'
        with open(f_name,"w") as f:
            f.write(page_py.text)
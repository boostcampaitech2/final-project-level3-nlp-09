from QA_model import inference

inf = inference.ExtractivedQAMdoel('QA_model/data/contexts')
inf.set_context('행성')
inf.set_question('이것의 이름은?')
inf.prepare_dataset()
print(inf.run_mrc())
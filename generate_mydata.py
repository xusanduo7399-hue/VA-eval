# generate_batch.py
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
import csv
import os

MODEL_FILE = 'C:/Users/kikyo/ParlAI/data/models/blender/blender_90M/model'
DATA_FILE = 'C:/Users/kikyo/ParlAI/data/mydata_parlai.txt'
OUT_CSV   = 'C:/Users/kikyo/ParlAI/data/blender90_preds.csv'
OUT_TXT   = 'C:/Users/kikyo/ParlAI/data/blender90_answers.txt'

def main():
    parser = ParlaiParser(True, True)
    opt = parser.parse_args([
        '-m', 'transformer/generator',
        '-mf', MODEL_FILE,
        '-t', 'fromfile:parlaiformat',
        '--fromfile-datapath', DATA_FILE,
        '--datatype', 'test',          # 不打乱顺序
        '--batchsize', '1',            # 逐条生成，保证与输入一一对应
        '--skip-generation', 'false',  # 确保生成文本
    ])

    agent = create_agent(opt, requireModelExists=True)
    world = create_task(opt, agent)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fcsv = open(OUT_CSV, 'w', encoding='utf-8', newline='')
    writer = csv.writer(fcsv)
    writer.writerow(['text', 'label', 'prediction'])

    ftxt = open(OUT_TXT, 'w', encoding='utf-8')

    idx = 0
    while not world.epoch_done():
        world.parley()
        acts = world.acts  # [teacher_msg, agent_msg]
        user_text = acts[0].get('text', '')
        gold = ''
        if 'labels' in acts[0]:
            labs = acts[0]['labels']
            gold = labs[0] if isinstance(labs, list) and labs else ''
        elif 'eval_labels' in acts[0]:
            labs = acts[0]['eval_labels']
            gold = labs[0] if isinstance(labs, list) and labs else ''

        pred = acts[1].get('text', '')

        writer.writerow([user_text, gold, pred])
        ftxt.write(f'{pred}\n')
        idx += 1

    fcsv.close()
    ftxt.close()
    print(f'Done. wrote {idx} examples to')
    print(f'  {OUT_CSV}')
    print(f'  {OUT_TXT}')

if __name__ == '__main__':
    main()

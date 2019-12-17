import json
import os
from os.path import isfile, dirname, realpath, join
import time
import subprocess

DATA_ROOT       = realpath(join(dirname(__file__), 'data'))
MODEL_PREFIX    = 'model-'
EVAL_PREFIX     = 'evaluation-'

CNMOW_CMD_TEMPLATE = """python3 evaluate_word2mat.py   \
    --encoders data/{model_prefix}{name}/mode:random-w2m_type:cnmow-word_emb_dim:784-.encoder   \
    --word_vocab data/{model_prefix}{name}/mode:random-w2m_type:cnmow-word_emb_dim:784-.vocab   \
    --outputdir data/{eval_prefix}{name}   \
    --outputmodelname {name}  \
    --downstream_eval full \
    --output_file=data/{eval_prefix}{name}/evaluation.csv \
    --downstream_tasks SICKEntailment STS15 STS16 MRPC Tense SubjNumber BigramShift CoordinationInversion OddManOut ObjNumber TREC Length Depth"""

HYBRID_CMD_TEMPLATE = """python3 evaluate_word2mat.py   \
    --encoders data/{model_prefix}{name}/mode:random-w2m_type:hybrid-word_emb_dim:400-.encoder   \
    --word_vocab data/{model_prefix}{name}/mode:random-w2m_type:hybrid-word_emb_dim:400-.vocab   \
    --outputdir data/{eval_prefix}{name}   \
    --outputmodelname {name}  \
    --downstream_eval full \
    --output_file=data/{eval_prefix}{name}/evaluation.csv \
    --downstream_tasks SICKEntailment STS15 STS16 MRPC Tense SubjNumber BigramShift CoordinationInversion OddManOut ObjNumber TREC Length Depth"""

def main():
    assert os.path.isdir(DATA_ROOT)

    evals = []
    models = []
    for f in os.listdir(DATA_ROOT):
        if f.startswith(MODEL_PREFIX):
            models.append(f[len(MODEL_PREFIX):])
        if f.startswith(EVAL_PREFIX):
            evals.append(f[len(EVAL_PREFIX):])

    missing_evals = set(models) - set(evals)

    for name in missing_evals:
        template = CNMOW_CMD_TEMPLATE
        if 'hybrid' in name:
            template = HYBRID_CMD_TEMPLATE

        cmd = template.format(model_prefix=MODEL_PREFIX, eval_prefix=EVAL_PREFIX, name=name).split(' ')
        cmd = [c for c in cmd if c]

        t0 = time.time()
        subprocess.check_call(cmd)
        elapsed = time.time() - t0

        print('----- completed variant: {}, took {:.2f}s'.format(name, elapsed))



if __name__ == '__main__':
    main()

import json
import os
from os.path import isfile, dirname, realpath, join
import time
import subprocess

from notify import SlackNotifier

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

def has_file_with_extension(dirname, ext):
    for f in os.listdir(join(DATA_ROOT, dirname)):
        if f.endswith(ext):
            return True
    return False


def main():
    assert os.path.isdir(DATA_ROOT)

    notifier = SlackNotifier()
    evals = []
    models = []

    # There may be empty 'model' or 'evaluation' folders, so we double-check
    for f in os.listdir(DATA_ROOT):
        if f.startswith(MODEL_PREFIX):
            if has_file_with_extension(f, '.encoder'):
                models.append(f[len(MODEL_PREFIX):])
        elif f.startswith(EVAL_PREFIX):
            if has_file_with_extension(f, 'evaluation.csv'):
                evals.append(f[len(EVAL_PREFIX):])

    missing_evals = sorted(set(models) - set(evals))
    print('Missing evaluations:')
    confirmed = []
    for name in missing_evals:
        print('- {}'.format(name))

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
        notifier.notify('DLNLP evaluation: completed variant: {}, took {:.2f}s'.format(name, elapsed))



if __name__ == '__main__':
    main()

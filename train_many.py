"""
Convenience script to train many variants.
"""

import json
import os
from os.path import isfile, dirname, realpath, join
import time
import subprocess

from notify import SlackNotifier

DATA_ROOT       = realpath(join(dirname(__file__), 'data'))
DATASET_1P      = join(DATA_ROOT, 'webbase_tiny')
DATASET_10P     = join(DATA_ROOT, 'webbase_mini')
DATASET_100P    = join(DATA_ROOT, 'webbase_extracted')
DOCS_COUNT_1P   = 1344426
DOCS_COUNT_10P  = 13444268
DOCS_COUNT_100P = 134442680

COMMON_ARGS = {
    'batch_size': 1024,
    'optimizer': 'adam,lr=0.0003',
    'max_words': 30000,
    'n_epochs': 1000,
    'n_negs': 20,
    'validation_frequency': 1000,
    'mode': 'random',
    'num_samples_per_item': 30,
    'patience': 10,
    'downstream_eval': 'none',  # Evaluation run separately from training
    'outputmodelname': ['mode', 'w2m_type', 'word_emb_dim'],
    'validation_fraction': 0.0001,
    'context_size': 5,
    'temp_path': '/tmp',
    'num_workers': 2,
    'stop_criterion': 'train_loss',
    'initialization': 'identity',
}
COMMON_ARGS_10P = dict(COMMON_ARGS, **{
    'dataset_path': DATASET_10P,
    'num_docs': DOCS_COUNT_10P,
})

VARIANTS = {
    'cbow-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cbow',
        'word_emb_dim': 784,
    }),
    'cmow-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cmow',
        'word_emb_dim': 784,
    }),
    'hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'word_emb_dim': 400,
    }),
    'hybrid-alpha16-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'word_emb_dim': 400,
        'explore_par': 16,
    }),
    'cnmow1-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 1,
    }),
    'cnmow1b-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 101,
    }),
    'cnmow2-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 2,
    }),
    'cnmow2b-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 201,
    }),
    'cnmow3-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 3,
        '_lambda': 0.5,
    }),
    'cnmow4-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 4,
        '_lambda': 0.5,
    }),
    'cnmow5-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 5,
    }),
    'cnmow6-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 6,
        'seed': 42,
    }),

    'cnmow7-784-10p': dict(COMMON_ARGS_10P, **{
       'w2m_type': 'cnmow',
       'word_emb_dim': 784,
       'cnmow_version': 7,
        '_lambda': 0.5,
    }),
    'cnmow8-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 8,
    }),
    'cnmow9-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 9,
    }),

    'cnmow3-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 3,
        '_lambda': 0.5,
    }),
    'cnmow4-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 4,
        '_lambda': 0.5,
    }),
    'cnmow5-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 5,
    }),
    'cnmow6-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 5,
    }),
    'cnmow7-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 7,
        '_lambda': 0.5,
    }),
    'cnmow8-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 8,
        '_lambda': 0.5,
    }),
    'cnmow9-hybrid-800-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'hybrid',
        'hybrid_cmow': 'cnmow',
        'word_emb_dim': 400,
        'cnmow_version': 9,
        '_lambda': 0.5,
    }),
    'cnmow6b-784-10p': dict(COMMON_ARGS_10P, **{
        'w2m_type': 'cnmow',
        'word_emb_dim': 784,
        'cnmow_version': 601,
    }),
}


def main():
    assert os.path.isdir(DATA_ROOT)

    notifier = SlackNotifier()

    for variant_name, _args in VARIANTS.items():
        print('----- Starting variant: {}'.format(variant_name))
        output_dir = join(DATA_ROOT, 'model-' + variant_name)
        output_file = join(output_dir, variant_name + '.csv')
        os.makedirs(output_dir, exist_ok=True)

        found_existing = False
        for f in os.listdir(output_dir):
            if f.endswith('.encoder'):
                found_existing = True
                break
        if found_existing:
            print('-----/ Variant already trained, skipping: {}'.format(variant_name))
            continue

        args = dict(_args, **{
            'outputdir': output_dir,
            'output_file': output_file,
        })
        flags = []
        for k, v in args.items():
            flags.append('--' + k)
            if isinstance(v, (list, tuple)):
                flags += v
            else:
                flags.append(str(v))

        t0 = time.time()
        subprocess.check_call(['python3', 'train_cbow.py'] + flags)
        elapsed = time.time() - t0
        # TODO: how to get this?
        epoch_count = -1

        metadata_fname = join(output_dir, 'metadata.csv')
        with open(metadata_fname, 'w') as f:
            f.write('Variant name, Docs count, Training time, Epoch count, Arguments\n')
            f.write('{}, {}, {}, {}, {}'.format(
                variant_name,
                args['num_docs'],
                elapsed,
                epoch_count,
                json.dumps(json.dumps(args))
            ))

        print('----- completed variant: {}, took {:.2f}s'.format(variant_name, elapsed))
        notifier.notify('DLNLP training: completed variant: {}, took {:.2f}s'.format(variant_name, elapsed))


if __name__ == '__main__':
    main()

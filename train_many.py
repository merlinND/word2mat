"""
Convenience script to train many variants.
"""

import json
import os
from os.path import isfile, dirname, realpath, join
import time
import subprocess

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
    'downstream_eval': 'full',
    'outputmodelname': 'mode w2m_type word_emb_dim',
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
}


def main():
    assert os.path.isdir(DATA_ROOT)

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


        args = dict(_args, **{
            'outputdir': output_dir,
            'output_file': output_file,
        })
        flags = sum([['--' + k, str(v)] for k, v in args.items()], [])

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



if __name__ == '__main__':
    main()

DLNLP Course project
====================

Reproducing results from CMOW
-----------------------------

When checking out SentEval, one must use the fix from [PR #52](https://github.com/facebookresearch/SentEval/pull/52), otherwise there will be an error in the SNLI evaluation task.


Training the hybrid model (CBOW + CMOW):
```
python train_cbow.py --w2m_type hybrid --batch_size=1024 --outputdir=data/models --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --temp_path /tmp --dataset_path=data/webbase_small --num_workers 2 --output_file data/models/hybrid.csv --num_docs 134442680 --stop_criterion train_loss --initialization identity
```

Evaluating performance of the hybrid model:
```
python3 evaluate_word2mat.py   \
    --encoders data/models/mode:random-w2m_type:hybrid-word_emb_dim:400-.encoder   \
    --word_vocab data/models/mode:random-w2m_type:hybrid-word_emb_dim:400-.vocab   \
    --outputdir data/evaluation   \
    --outputmodelname hybrid  \
    --downstream_eval full \
    --output_file=data/evaluation/hybrid.csv
    #--downstream_tasks SNLI
```

Training the CNMOW model
------------------------

First, select the variant by commenting out the other lines in `word2mat.py` (`_continual_multiplication_nn`).

Train on the reduced dataset:

```
python3 train_cbow.py --w2m_type cnmow --batch_size=1024 --outputdir=data/model-cnmow-01 --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --temp_path /tmp --dataset_path=data/webbase_mini --num_workers 2 --output_file data/models/cnmow-01.csv --num_docs 13444268 --stop_criterion train_loss --initialization identity
```
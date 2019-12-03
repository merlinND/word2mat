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

Evaluating performance of the CBOW model:
```
python3 evaluate_word2mat.py   \
    --encoders data/model-cbow-784-10p/random-w2m_type:cbow-word_emb_dim:784-.encoder   \
    --word_vocab data/model-cbow-784-10p/random-w2m_type:cbow-word_emb_dim:784-.vocab   \
    --outputdir data/evaluation   \
    --outputmodelname cbow-784-10p  \
    --downstream_eval full \
    --output_file=data/evaluation-cbow-784-10p/evaluation.csv
    #--downstream_tasks SNLI
```

Training the CNMOW model
------------------------

First, select the variant by commenting out the other lines in `word2mat.py` (`_continual_multiplication_nn`).

Train on the reduced dataset:

```
python3 train_cbow.py --w2m_type cnmow --batch_size=1024 --outputdir=data/model-cnmow-01 --optimizer adam,lr=0.0003 --max_words=30000 --n_epochs=1000 --n_negs=20 --validation_frequency=1000 --mode=random --num_samples_per_item=30 --patience 10 --downstream_eval full --outputmodelname mode w2m_type word_emb_dim --validation_fraction=0.0001 --context_size=5 --word_emb_dim 400 --temp_path /tmp --dataset_path=data/webbase_mini --num_workers 2 --output_file data/models/cnmow-01.csv --num_docs 13444268 --stop_criterion train_loss --initialization identity
```

Evaluating performance of the newly trained model:
```
python3 evaluate_word2mat.py   \
    --encoders data/model-cnmow-01/mode:random-w2m_type:cnmow-word_emb_dim:400-.encoder   \
    --word_vocab data/model-cnmow-01/mode:random-w2m_type:cnmow-word_emb_dim:400-.vocab   \
    --outputdir data/evaluation-cnmow-01   \
    --outputmodelname cnmow  \
    --downstream_eval full \
    --output_file=data/evaluation-cnmow-01/cnmow.csv
    #--downstream_tasks SNLI
```


Tasks
-----

**Record training times**: runtime, number of sentences / sec, number of epochs before convergence

Train baselines on 10% dataset, and evaluate:

- CBOW
- CMOW
- Hybrid

Dimensionality of word embeddings: 784
Hybrid: 400 + 400.

Train our models:

- Hybrid (CMOW / CBOW) trained with exploration / exploitation: `params.explore_par` values: [1, 2, 10, 50]
- CNMOW1: ReLU between words, including the first word
- CNMOW2: ReLU between words, excluding the first word
- CNMOW3: weigthed skip connections, parameter lambda: [0.25, 0.5, 0.75, 1]
          0 means only keeping the current word, not adding the previous one; 1 means ignoring the current word.
- CNMOW4: ReLU nonlinearity (excluding the first word) and weigthed skip connections, parameter lambda: [0.25, 0.5, 0.75, 1]
- CNMOW5: weigthed skip connection with "learnt lambda"
          Lambda is predicted by multiplying the current word with a weight matrix, adding a bias, and passing through a sigmoid nonlinearity.
          Therefore there is one lambda value per entry of the embedding.
          /!\ TODO: make sure the weight matrix is being updated during training.
- CNMOW6: more like an RNN. At every step, multiply the result of the continuous multiplication by some shared weights, add shared bias, pass through ReLU.

Since training on CPU cluster is slow, we do hyperparameter selection on 1% of the data, then train the variant with best results on the 10% dataset to compare with the rest of the models.

Then, select the best ones and train again in Hybrid mode (with CBOW).

Selected evaluation tasks:

- Probing tasks (dim 400): WC, B Shift, CoordInv
- Supervised evaluation: TREC, STS-B, CR
- Unsupervised evaluation: STS15, STS16

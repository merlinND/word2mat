"""
This script is intended to provide an interface for standardized evaluation of encoders. For implementation, you need to provide the interface with a bunch of functions that you then need to pass to the run_and_evaluate function in the following
fashion:

>>> run_and_evaluate(run_experiment, get_params_parser, batcher, prepare).

For detailed descriptions of what these methods have to provide, please refer to the functions documentation.
"""

import torch, os, sys, logging, pickle
import numpy as np
import pandas as pd

from torch.autograd import Variable
from mutils import run_hyperparameter_optimization, write_to_csv
from warnings import warn

# Set PATHs to SentEval
PATH_SENTEVAL = '../SentEval/'
PATH_TO_DATA = '../SentEval/data'
assert os.path.exists(PATH_SENTEVAL) and os.path.exists(PATH_TO_DATA), "Set path to SentEval + data correctly!"

sys.path.insert(0, PATH_SENTEVAL)
import senteval

# Set to 1 or 0 to disable usage of parallel evaluation
PARALLEL_EVALUATION_CORES = os.cpu_count()
PARALLEL_EVALUATION_ENABLED = PARALLEL_EVALUATION_CORES > 1

def _get_score_for_name(downstream_results, name):
    if name in ["CR", "CoordinationInversion", "SST2", "Length", "OddManOut", "Tense", "SUBJ", "MRPC", "ObjNumber", "SubjNumber", "Depth", "WordContent", "SST5", "SNLI", "MPQA", "BigramShift", "MR", "TREC", "TopConstituents", "SICKEntailment"]:
        return downstream_results[name]["acc"]
    elif name in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
        all_results = downstream_results[name]["all"]
        spearman = all_results["spearman"]["mean"]
        pearson = all_results["pearson"]["mean"]
        return (spearman, pearson)
    elif name in ["STSBenchmark", "SICKRelatedness"]:
        spearman = downstream_results[name]["spearman"]
        pearson = downstream_results[name]["pearson"]
        return (spearman, pearson)
    else:
        warn("Can not extract score from downstream task " + name)
        return 0

def _run_experiment_and_save(run_experiment, params, batcher, prepare):

    encoder, losses = run_experiment(params)

    # Save encoder
    outputmodelname = construct_model_name(params.outputmodelname, params)
    torch.save(encoder, os.path.join(params.outputdir, outputmodelname + '.encoder'))

    # write training and validation loss to csv file
    with open(os.path.join(params.outputdir, outputmodelname + "_losses.csv"), 'w') as loss_csv:
        loss_csv.write("train_loss,val_loss\n")
        for train_loss, val_loss in losses:
            loss_csv.write(",".join([str(train_loss), str(val_loss)]) + "\n")

    scores = {}
    # Compute scores on downstream tasks
    if params.downstream_eval and params.downstream_eval != "none":
        downstream_scores = _evaluate_downstream_and_probing_tasks(encoder, params, batcher, prepare)

        # from each downstream task, only select scores we care about
        to_be_saved_scores = {}
        for score_name in downstream_scores:
            to_be_saved_scores[score_name] = _get_score_for_name(downstream_scores, score_name)
        scores.update(to_be_saved_scores)

    # Compute word embedding score
    if params.word_embedding_eval:
        output_path = _save_embeddings_to_word2vec(encoder, outputmodelname, params)

    # Save results to csv
    if params.output_file:
        write_to_csv(scores, params)

    return scores

def _load_embeddings_from_word2vec(fname):
    w = load_embedding(fname, format="word2vec", normalize=True, lower=True, clean_words=False)
    return w

def _save_embeddings_to_word2vec(encoder, outputmodelname, params):

    # Load lookup table as numpy
    embeddings = encoder.lookup_table
    embeddings = embeddings.weight.data.cpu().numpy()

    # Load (inverse) vocabulary to match ids to words
    path_to_vocabulary = os.path.join(params.outputdir, outputmodelname + '.vocab')
    vocabulary = pickle.load(open(path_to_vocabulary, "rb" ))[0]
    inverse_vocab = {vocabulary[w] : w for w in vocabulary}

    # Open file and write values in word2vec format
    output_path = os.path.join(params.outputdir, outputmodelname + '.emb')
    f = open(output_path, 'w')
    print(embeddings.shape[0] - 1, embeddings.shape[1], file = f)
    for i in range(1, embeddings.shape[0]): # skip the padding token
        cur_word = inverse_vocab[i]
        f.write(" ".join([cur_word] + [str(embeddings[i, j]) for j in range(embeddings.shape[1])]) + "\n")

    f.close()

    return output_path

def _evaluate_downstream_and_probing_tasks(encoder, params, batcher, prepare):
    # define senteval params
    eval_type = sys.argv[1] if len(sys.argv) > 1 else ""
    # TODO: do we need this?
    use_pytorch = not PARALLEL_EVALUATION_ENABLED

    if params.downstream_eval == "full":

        ## for comparable evaluation (as in literature)
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': use_pytorch, 'kfold': 10}
        params_senteval['classifier'] = {'nhid': params.nhid, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}

    elif params.downstream_eval == "test":
        ## for testing purpose
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': use_pytorch, 'kfold': 5}
        params_senteval['classifier'] = {'nhid': params.nhid, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
    else:
        sys.exit("Need to specify if you want to run it in full mode ('full', takes a long time," \
                "but is comparable to literature) or test mode ('test', fast, but not as accurate and comparable).")

    # Pass encoder and command line parameters
    if not PARALLEL_EVALUATION_ENABLED:
        params_senteval['word2mat'] = encoder
    params_senteval['cmd_params'] = params

    # evaluate
    if not PARALLEL_EVALUATION_ENABLED:
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        results = se.eval(params.downstream_tasks)
    else:
        import multiprocess

        def eval_one(task_name):
            se = senteval.engine.SE(params_senteval, batcher, prepare)
            results = se.eval(task_name)
            return results

        # Handle relatedness task separately because it uses CUDA
        handle_relatedness = False
        if 'SICKRelatedness' in params.downstream_tasks:
            handle_relatedness = True
            params.downstream_tasks.remove('SICKRelatedness')

        results = {}
        if len(params.downstream_tasks) >= 1:
            pool = multiprocess.Pool(min(len(params.downstream_tasks), PARALLEL_EVALUATION_CORES))
            par_results = pool.map(eval_one, params.downstream_tasks)
            results = dict(results, **dict(zip(params.downstream_tasks, par_results)))

        if handle_relatedness:
            try:
                se = senteval.engine.SE(params_senteval, batcher, prepare)
                extra_result = se.eval('SICKRelatedness')
                results['SICKRelatedness'] = extra_result
            except Exception as e:
                print('Error! Could not evaluate SICKRelatedness: ' + str(e))

    return results

def construct_model_name(names, params):
    """Constructs model name from all params in "name", unless name only holds a single argument."""
    if len(names) == 1:
        return names[0]
    else:
        # construct model name from configuration
        name = ""
        params_dict = vars(params)

        for key in names:

            name += str(key) + ":" + str(params_dict[key]) + "-"

    return name


def _add_common_arguments(parser):

    group = parser.add_argument_group("common", "Arguments needed for evaluation.")
    # paths
    parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory", required = True)
    parser.add_argument("--outputmodelname", type=str, nargs = "+", default=["mymodel"], help="If one argument is passed, the model is saved at the respective location. If multiple arguments are passed, these are interpreted of names of parameters from which the modelname is automatically constructed in a <key:value> fashion.", required = True)
    parser.add_argument('--output_file', type=str, default=None, help= \
        "Specify the file name to save the result in. Default: [None]")

    # hyperparameter opt
    parser.add_argument("--optimization", type=str, default=None)
    parser.add_argument("--optimization_spaces", type=str, default="default_optimization_spaces")
    parser.add_argument("--optimization_iterations", type=int, default=10)

    # reproducibility
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    # evaluation
    parser.add_argument("--downstream_eval", type=str, help="Whether to perform 'full'" \
                "downstream evaluation (slow), 'test' downstream evaluation (fast).",
                choices = ["test", "full", "none"], required = True)
    parser.add_argument("--downstream_tasks", type=str, nargs = "+",
                         default=['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                            'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                            'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                            'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'],
                         help="Downstream tasks to evaluate on.",
                        choices = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                            'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                            'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                            'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'])
    parser.add_argument("--word_embedding_eval", action="store_true", default=False, help="If specified," \
                "evaluate the word embeddings and store the results.")
    parser.add_argument("--nhid", type=int, default=0, help="Specify the number of hidden units used at test time to train classifiers. If 0 is specified, no hidden layer is employed.")


    return parser

def run_and_evaluate(run_experiment, get_params_parser, batcher, prepare):
    """
    Trains and evaluates one or several configurations, given some functionality to train the encoder
    and generate embeddings at test time. In particular, 4 functions have to be provided:

    - get_params_parser: () -> argparse.Argumentparser
            Creates and returns an argument parser with arguments that will be used by the
            run_experiment parser. Note that it must not contain any of the arguments that are
            are later added by the run_and_evaluate function, and are shared by all implementations.  Please refer to the _add_common_arguments function for an overview of common arguments.
            - outputdir is not optional and the dir has to exist

    - run_experiment: params -> encoder, losses
            The function that trains the encoder. As input, it gets the params dictionary that contains
            the arguments parsed from the command line. As output, it has to return the encoder object that will
            at test time be passed to the 'batcher' function.
            The encoder is required to have an attribute 'lookup_table' which is a torch.nn.Embedding.
            'losses' is a list of successive
            (train_loss, val_loss) pairs that will be written to a csv file. If you do not want to track
            the loss, simply return an empty list.
            - params need to include an 'outputmodelname', else the params need to satisfy construct_model_name

    - batcher: (params, batch) -> nparray of shape (batch_size, embedding_size)
            A function as defined by SentEval (see https://github.com/facebookresearch/SentEval).
            'params' is a dictionary that contains the encoder as "params['word2mat']", as well as
            further parameters specified by the implementer and added via the 'prepare' function.
            - the params namespace from run_experiments is available as params['cmd_params']

    - prepare: (senteval_params, samples) -> ()
            A function as defined by SentEval (see https://github.com/facebookresearch/SentEval).
            In particular, this function can be used to add further arguments to the senteval_params
            dictionary for use in batcher.
            'senteval_params' is a dictionary that in turn contains the 'params' objective parsed from
            the command line as the argument 'cmd_params'.
    """
    parser = get_params_parser()
    parser = _add_common_arguments(parser)
    params = parser.parse_args()

    os.makedirs(params.outputdir, exist_ok=True)

    if params.optimization:
        def hyperparameter_optimization_func(opts):
            results = _run_experiment_and_save(run_experiment, opts, batcher, prepare)
            return results["CR"]
        run_hyperparameter_optimization(params, hyperparameter_optimization_func)
    else:
        _run_experiment_and_save(run_experiment, params, batcher, prepare)


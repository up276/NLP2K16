import tensorflow as tf
import reader
from sklearn.manifold import TSNE
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import spatial
import matplotlib
import pylab as plt
matplotlib.use('Agg')


# DEFINING FLAGS

flags = tf.flags
flags.DEFINE_string('data_path','../simple-examples/data','Data path to load the data')
flags.DEFINE_string('checkpoint_path','../results/runs_lstm/1478206924','Directory with model check point')
FLAGS = flags.FLAGS



# This function receives two wordsm get the embedding and finds the cosine similarity score between those two words
def similarity(model,word1,word2):
     emb = tf.get_default_graph().get_tensor_by_name("Model/embedding:0")
     tensr = tf.convert_to_tensor([wordLookup(word1),wordLookup(word2)],name='embeddings')
     input_for_cosine = tf.nn.embedding_lookup(emb,tensr).eval()
     cosine_distance = cosine_similarity(input_for_cosine[0],input_for_cosine[1])
     return cosine_distance



# This function checks if the word is present in vocab. Returns none if not present.
def wordLookup(input_word,train_path=os.path.join(FLAGS.data_path,'ptb.train.txt')):
    vocab = reader._build_vocab(train_path)
    if input_word in vocab:
        return vocab[input_word]
    else:
        print("Word is not present in vocab")
        return None

#Scoring rubric. "," is not present hence commenting it
def score(model):
    score = 0.
    score += similarity(model,'a', 'an') > similarity(model,'a', 'document')
    score += similarity(model,'in', 'of') > similarity(model,'in', 'picture')
    score += similarity(model,'nation', 'country') > similarity(model,'nation', 'end')
    score += similarity(model,'films', 'movies') > similarity(model,'films', 'almost')
    score += similarity(model,'workers', 'employees') > similarity(model,'workers', 'movies')
    score += similarity(model,'institutions', 'organizations') > similarity(model,'institution', 'big')
    score += similarity(model,'assets', 'portfolio') > similarity(model,'assets', 'down')
    #score += similarity(model,"'", ",") > similarity(model,"'", 'quite')
    score += similarity(model,'finance', 'acquisition') > similarity(model,'finance', 'seems')
    score += similarity(model,'good', 'great') > similarity(model,'good', 'minutes')
    return score

#returns t-sne representation
def getTSNE(model):
    print("Running TSNE on word embeddings")
    embedding = tf.get_default_graph().get_tensor_by_name("Model/embedding:0").eval()
    TSNE_obj = TSNE(learning_rate=300)
    start_time = time.time()
    tsne_transform = TSNE_obj.fit_transform(embedding)
    end_time = time.time()
    print("Finished in %s"%(end_time-start_time))
    return tsne_transform.T

def vizualizeTSNE(tsneEmbedding,path_to_save,path_to_train=os.path.join(FLAGS.data_path,'ptb.train.txt'),samples=400):
    """
    Makes visualization of random sample of t-SNE embedding and annotate with words, saves.
    """

    vocab = reader._build_vocab(path_to_train)
    reverse_vocab = {v:k for k,v in vocab.iteritems()}
    random_ix = np.random.choice(tsneEmbedding.T.shape[0], samples)
    embadding_random = tsneEmbedding.T[random_ix].T
    keys = np.array(reverse_vocab.keys())[random_ix]
    plt.figure(figsize=(25,25))
    plt.scatter(embadding_random[0],embadding_random[1])
    plt.title("TSNE Word Representation from random %s words "%samples)

    for i, txt in enumerate(embadding_random.T):
        plt.annotate(reverse_vocab[keys[i]], (embadding_random[0][i],embadding_random[1][i]))
        plt.savefig(path_to_save)

def main(_):
    if not FLAGS.checkpoint_path:
        print("Checkpoint path is missing ... !!!")
    else:
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        print("Loaded Model From %s"%FLAGS.checkpoint_path)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            model = tf.get_default_graph()
            print('Score : %s out of 9'%score(model))
            print("Starting TSNE Visualization")
            tSNE_data_to_visualize = getTSNE(model)
            np.savetxt(os.path.join(FLAGS.checkpoint_path,'tsne_proj.txt'),tSNE_data_to_visualize)
            vizualizeTSNE(tSNE_data_to_visualize,os.path.join(FLAGS.checkpoint_path,'tsne.png'),samples=500)
            print("TSNe visualization is done. Plot has been saved")

if __name__=="__main__":
    tf.app.run()

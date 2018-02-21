import argparse

import numpy as np

import skip_gram


def learnVocabFromTrainFile(train, mincount):
    tmpwords = list()
    vocab = list()
    vocab_hash = dict()
    words = list()
    vocab.append(['</s>', 0])
    vocab_hash['</s>'] = 0
    with open(train) as f:
        for line in f:
            sp = line.strip().split()
            for word in sp:
                tmpwords.append(word)
                if word not in vocab_hash:
                    vocab.append([word, 0])
                    vocab_hash[word] = len(vocab_hash)
                vocab[vocab_hash[word]][1] += 1
            tmpwords.append('</s>')
            vocab[vocab_hash['</s>']][1] += 1
    vocab[1:] = sorted(vocab[1:], key=lambda x: x[1], reverse=True)
    vocab_hash = dict()
    while vocab[len(vocab) - 1][1] < mincount and len(vocab) > 1:
        vocab.pop()
    for voc in vocab:
        vocab_hash[voc[0]] = len(vocab_hash)
    for word in tmpwords:
        if word in vocab_hash:
            words.append(vocab_hash[word])
    vocab_count = list()
    for voc in vocab:
        vocab_count.append(voc[1])
    vocab_count = np.asarray(vocab_count, dtype=np.int64)
    words = np.asarray(words, dtype=np.int64)
    print('vocab size', len(vocab))
    print('words in train file', len(words))
    return vocab, vocab_hash, words, vocab_count


def initUnigramTable(vocab):
    train_words_pow = 0
    for voc in vocab:
        train_words_pow += voc[1] ** 0.75
    i = 0
    d1 = vocab[i][1] ** 0.75 / train_words_pow
    table = np.zeros((100000000), dtype=np.int64)
    for a in range(100000000):
        table[a] = i
        if a / 100000000 > d1:
            i += 1
            d1 += (vocab[i][1] ** 0.75 / train_words_pow)
        if i >= 100000000:
            i = 99999999
    return table


def trainModel(train, output, index, size, window, sample, negative, threads, iters, mincount, alpha):
    print('reading file', train)
    vocab, vocab_hash, words, vocab_count = learnVocabFromTrainFile(
        train, mincount)
    syn0 = np.asarray(np.random.random(
        (len(vocab), size)) - 0.5, dtype=np.float32) / size
    syn1 = np.zeros((len(vocab), size), dtype=np.float32)
    print('init the unigram table')
    table = initUnigramTable(vocab)
    print('start training')
    skip_gram.train(size, alpha, window, sample, negative,
                    threads, iters, mincount, syn0, syn1, words, vocab_count, table)
    np.save(output + '.npy', syn0)
    with open(index, 'w') as f:
        for i in range(len(vocab)):
            f.write(str(i) + ' ' + vocab[i][0] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train', help='Use text data from this file to train the model')
    parser.add_argument(
        'output', help='Use this file to save the resulting word vectors(numpy format)')
    parser.add_argument('index', help='Use this file to save word indexes')
    parser.add_argument('--size', type=int, default=100,
                        help='Size of word vectors')
    parser.add_argument('--window', type=int, default=5,
                        help='Max skip length between words')
    parser.add_argument('--sample', type=float, default=0.001,
                        help='Threshold for occurence of words. Those that appear with higher frequency in the training data will be randomly down-sampled')
    parser.add_argument('--negative', type=int, default=5,
                        help='Number of negative examples')
    parser.add_argument('--threads', type=int, default=12,
                        help='Number of threads')
    parser.add_argument('--iters', type=int, default=5, help='Training iters')
    parser.add_argument('--mincount', type=int, default=5,
                        help='Discard words that appear less than mincount times')
    parser.add_argument('--alpha', type=float, default=0.025,
                        help='Starting learning rate')
    args = parser.parse_args()
    trainModel(args.train, args.output, args.index, args.size, args.window,
               args.sample, args.negative, args.threads, args.iters, args.mincount, args.alpha)

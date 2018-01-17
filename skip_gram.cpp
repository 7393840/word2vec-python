#include <iostream>
#include <cmath>
#include <vector>
#include <thread>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
using namespace std;
namespace py = boost::python;
class skipgram{
public:
    skipgram(int64_t _embsize, float _alpha, int64_t _window, float _sample, int64_t _negative, int64_t _num_threads, int64_t _iter, int64_t _mincount, py::numpy::ndarray _syn0, py::numpy::ndarray _syn1, py::numpy::ndarray _words, py::numpy::ndarray _vocabcount, py::numpy::ndarray _table): 
        embsize(_embsize), alpha(_alpha), window(_window), sample(_sample), negative(_negative), num_threads(_num_threads), iter(_iter), word_count_actual(0), starting_alpha(_alpha), expTable(1000 + 1, 0), syn0_(_syn0), syn1_(_syn1), words_(_words), vocabcount_(_vocabcount), table_(_table) {
        for (int64_t i = 0; i < 1000; i++) {
            expTable[i] = exp((i / (float)1000 * 2 - 1) * 6);
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
        syn0 = (float*)(_syn0.get_data());
        syn1 = (float*)(_syn1.get_data());
        words = (int64_t*)(_words.get_data());
        vocabcount = (int64_t*)(_vocabcount.get_data());
        table = (int64_t*)(_table.get_data());
    }
    int64_t embsize = 100;
    float alpha = 0.025;
    int64_t window = 5;
    float sample = 1e-3;
    int64_t negative = 5;
    int64_t num_threads = 12;
    int64_t iter = 5;
    int64_t word_count_actual;
    float starting_alpha;
    vector<float> expTable;
    float* syn0;
    float* syn1;
    int64_t* words;
    int64_t* vocabcount;
    int64_t* table;
    py::numpy::ndarray syn0_;
    py::numpy::ndarray syn1_;
    py::numpy::ndarray words_;
    py::numpy::ndarray vocabcount_;
    py::numpy::ndarray table_;
    clock_t start;
    void train() {
        start = clock();
        vector<thread> threads;
        for(int64_t i = 0; i < num_threads; i++) {
            threads.push_back(thread(bind(&skipgram::threadfunc, this, i)));
        }
        for(int64_t i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }
    void threadfunc(int64_t id) {
        uint64_t next_random = id;
        vector<float> neu1e(embsize);
        int64_t wstart = words_.shape(0) * id / num_threads;
        int64_t wend = words_.shape(0) * (id + 1) / num_threads;
        for(int64_t it = 0; it < iter; it++) {
            int64_t wcurrent = wstart;
            int64_t acc = 0;
            while(wcurrent < wend){
                int64_t word;
                vector<int64_t> sentence;
                while (true) {
                    if(wcurrent == wend) {
                        word_count_actual += acc;
                        acc = 0;
                        break;
                    }
                    word = words[wcurrent];
                    wcurrent++;
                    if (word == 0) {
                        break;
                    }
                    if (sample > 0) {
                        float ran = (sqrt(vocabcount[word] / (sample * words_.shape(0))) + 1) * (sample * words_.shape(0)) / vocabcount[word];
                        next_random = next_random * (uint64_t)25214903917 + 11;
                        if (ran >= (next_random & 0xFFFF) / (float)65536) {
                            sentence.push_back(word);
                        } else {
                            acc++;
                        }
                    }
                }
                for(int64_t sentence_position = 0; sentence_position < (int64_t)sentence.size(); sentence_position++) {
                    word = sentence[sentence_position];
                    next_random = next_random * (uint64_t)25214903917 + 11;
                    int64_t b = next_random % window;
                    for (int64_t a = b; a < window * 2 + 1 - b; a++) {
                        if (a != window) {
                            int64_t c = sentence_position - window + a;
                            if (c < 0 || c >= (int64_t)sentence.size()) {
                                continue;
                            }
                            int64_t last_word = sentence[c];
                            int64_t l1 = last_word * embsize;
                            for (int64_t c = 0; c < embsize; c++) {
                                neu1e[c] = 0;
                            }
                            for (int64_t d = 0; d < negative + 1; d++) {
                                int64_t target, label;
                                if (d == 0) {
                                    target = word;
                                    label = 1;
                                } else {
                                    next_random = next_random * (uint64_t)25214903917 + 11;
                                    target = table[(next_random >> 16) % table_.shape(0)];
                                    if (target == 0) {
                                        target = next_random % (vocabcount_.shape(0) - 1) + 1;
                                    }
                                    if (target == word) {
                                        continue;
                                    }
                                    label = 0;
                                }
                                int64_t l2 = target * embsize;
                                float f = 0;
                                float g = 0;
                                for (int64_t c = 0; c < embsize; c++) {
                                    f += syn0[c + l1] * syn1[c + l2];
                                }
                                if (f > 6) {
                                    g = (label - 1) * alpha;
                                } else if (f < -6) {
                                    g = (label - 0) * alpha;
                                } else {
                                    g = (label - expTable[(int64_t)((f + 6) * (1000 / 6 / 2))]) * alpha;
                                }
                                for (int64_t c = 0; c < embsize; c++) {
                                    neu1e[c] += g * syn1[c + l2];
                                }
                                for (int64_t c = 0; c < embsize; c++) {
                                    syn1[c + l2] += g * syn0[c + l1];
                                }
                            }
                            for (int64_t c = 0; c < embsize; c++) {
                                syn0[c + l1] += neu1e[c];
                            }
                        }
                    }
                    acc++;
                    if (acc > 10000) {
                        word_count_actual += acc;
                        acc = 0;
                        clock_t now = clock();
                        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, word_count_actual / (float)(iter * words_.shape(0) + 1) * 100, word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
                        fflush(stdout);
                        alpha = starting_alpha * (1 - word_count_actual / (float)(iter * words_.shape(0) + 1));
                        if (alpha < starting_alpha * 0.0001) {
                            alpha = starting_alpha * 0.0001;
                        }
                    }
                }
            }
        }
    }
};
void train(int64_t _embsize, float _alpha, int64_t _window, float _sample, int64_t _negative, int64_t _num_threads, int64_t _iter, int64_t _mincount, py::numpy::ndarray _syn0, py::numpy::ndarray _syn1, py::numpy::ndarray _words, py::numpy::ndarray _vocabcount, py::numpy::ndarray _table) {
    skipgram sg(_embsize, _alpha, _window, _sample, _negative, _num_threads, _iter, _mincount, _syn0, _syn1, _words, _vocabcount, _table);
    sg.train();
}
BOOST_PYTHON_MODULE(skip_gram) {
    py::numpy::initialize();
    py::def("train", &train);
}

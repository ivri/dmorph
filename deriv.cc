#include "deriv.h"
#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>

#include "cnn/gru.h"

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;
using namespace boost::filesystem;

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

cnn::Dict d, dc;
int kSOS, kEOS, kSOW, kEOW, kUNK;

//parameters
unsigned LAYERS = 3;
unsigned EMBEDDING_DIM = 200;
unsigned EMBEDDING_CHAR_DIM = 15;
unsigned HIDDEN_DIM = 200;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned INPUT_LEX_SIZE = 0;

template <class rnn_t> int main_body(variables_map vm);
void initialise(Model &model, const string &filename);
void ReadFile(const char* fname, vector<VocabEntryPtr>& dataset);


int main(int argc, char **argv)
{
	cnn::initialize(argc, argv);

	// command line processing
	variables_map vm;
	options_description opts("Allowed options");
	opts.add_options()
		("help", "print help message")
		("input,i", value<string>(), "file containing training sentences. Either a single file with "
		 "each line consisting of source ||| target ||| align; or ")
		("devel,d", value<string>(), "file containing development sentences (see --input)")
		("test", value<string>(), "file containing test sentences (see -- input)")
		("initialise", value<string>(), "file containing the saved model parameters")
		("sup-train", "run training of the model")
		("threshold-src,s", value<int>(), "keep only the <num> most frequent words (source)")
		("treport,r", value<int>(), "report training every i iterations")
		("dreport,R", value<int>(), "report dev every i iterations")
		("batch_size", value<int>(), "batch size in unsupervised learning")
		("batch_iter", value<int>(), "max batch iterations in unsup learning")
		("epochs,e", value<int>(), "max number of epochs")
		("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
		("embedding,E", value<int>()->default_value(EMBEDDING_DIM), "use <num> dimensions for word embeddings")
		("part-embedding,P", value<int>()->default_value(EMBEDDING_CHAR_DIM), "use <num> dimensions for character embeddings")
		("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
		("words,w", value<string>(), "Pretrained word embeddings, EMBEDDING DIM should be the same.")
		("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
		("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
		("decode", "decode sentences in the test set")
		("nobase", "don't include base form")
		("singledir", "use single left and right context direction (towards central word)")
		;
	store(parse_command_line(argc, argv, opts), vm);

	notify(vm);

	bool valid_command=vm.count("sup-train") || vm.count("decode");

	if (vm.count("help") || (! valid_command ) ) {
		cout << opts << "\n";
		return 1;
	}

	if (vm.count("lstm")) {
		cout << "%% Using LSTM recurrent units" << endl;
		return main_body<LSTMBuilder>(vm);
	} else if (vm.count("gru")) {
		cout << "%% Using GRU recurrent units" << endl;
		return main_body<GRUBuilder>(vm);
	} else {
		cout << "%% Using Simple RNN recurrent units" << endl;
		return main_body<SimpleRNNBuilder>(vm);
	}
}

template <class rnn_t>
int main_body(variables_map vm)
{
	unsigned MAX_EPOCH = 100;
	unsigned WRITE_EVERY_I = 1000;
	unsigned report = 1000;

	unsigned batch_size = 10;
	unsigned batch_iter = 2;
	unordered_map<unsigned, vector<float>> pretrained;
	
	kSOS = d.convert("<s>");
	kEOS = d.convert("</s>");
	kSOW = dc.convert("{");
	kEOW = dc.convert("}");
	
	bool nobase, singledir = false;

	//----
	if (vm.count("batch_size")) batch_size = vm["batch_size"].as<int>();
	if (vm.count("batch_iter")) batch_iter = vm["batch_iter"].as<int>();

	if (vm.count("dreport")) WRITE_EVERY_I = vm["dreport"].as<int>();
	if (vm.count("treport")) report = vm["treport"].as<int>();
	if (vm.count("epochs")) MAX_EPOCH = vm["epochs"].as<int>();

	if (vm.count("layers")) LAYERS = vm["layers"].as<int>();
	if (vm.count("embedding")) EMBEDDING_DIM = vm["embedding"].as<int>();
	if (vm.count("hidden")) HIDDEN_DIM = vm["hidden"].as<int>();
	if (vm.count("part-embedding")) EMBEDDING_CHAR_DIM = vm["part-embedding"].as<int>();

        if (vm.count("nobase")) nobase = true;
        if (vm.count("singledir")) singledir = true;

	// ---- read training sentences
	vector<VocabEntryPtr> training;
	Model model;

	if (vm.count("words")) {
    		cerr << "Loading from " << vm["words"].as<string>() << " with" << EMBEDDING_DIM << " dimensions\n";
    		ifstream in(vm["words"].as<string>().c_str());
    		string line;
    		getline(in, line);
    		vector<float> v(EMBEDDING_DIM, 0);
    		string word;
		auto c = 0;
    		while (getline(in, line)) {
      			istringstream lin(line);
      			lin >> word;
      			for (unsigned i = 0; i < EMBEDDING_DIM; ++i) lin >> v[i];
      			unsigned id = d.convert(word);
      			pretrained[id] = v;
			++c;
			if (c==300000) break;
    		}
  	}

	d.freeze();
	d.set_unk("UNK");
        kUNK = d.convert("UNK");
	pretrained[kUNK] = vector<float>(EMBEDDING_DIM, 0);
	
	if  (vm.count("input")) {
		if (! exists(vm["input"].as<string>())) {
			cout << "the input file doesnt exist" << endl;
			return 1;
		}
		ReadFile(vm["input"].as<string>().c_str(), training);
	}

	dc.freeze();

	INPUT_VOCAB_SIZE = d.size();
	INPUT_LEX_SIZE = dc.size();

	if (vm.count("sup-train")) {
		vector<VocabEntryPtr> devel;

		if (vm.count("devel")) {
			if (! exists(vm["devel"].as<string>())) {
				cout << "the input file doesnt exist" << endl;
				return 1;
			}
			ReadFile(vm["devel"].as<string>().c_str(), devel);
		}

		// ---- output vocab, corpus stats
		cout << "%% Training has " << training.size() << " sentence pairs\n";
		cout << "%% Development has " << devel.size() << " sentence pairs\n";
		cout << "%% source vocab " << INPUT_VOCAB_SIZE << " unique words\n";

		cout << "%% batch size in unsupervised learning: " << batch_size << endl;
		cout << "%% batch max iterations in unsup learning:" << batch_iter << endl;

		//---- file name for saving the parameters
		ostringstream os;
		os << "lm"
			<< '_' << LAYERS
			<< "_w" << EMBEDDING_DIM
			<< "_h" << HIDDEN_DIM
			<< "_c"<< EMBEDDING_CHAR_DIM
			<< '_' << ((vm.count("lstm")) ? "lstm" : (vm.count("gru")) ? "gru" : "rnn")
			<< "-pid" << getpid() << ".params";
		const string fname = os.str();
		cerr << "Parameters will be written to: " << fname << endl;

		cerr << "%% layers " << LAYERS << " embedding " << EMBEDDING_DIM << " hidden " << HIDDEN_DIM << endl;
		cerr << "%%  Character embedding dimensionality is set to " << EMBEDDING_CHAR_DIM << endl;


		bool use_momentum = false;
		Trainer* sgd = nullptr;
		if (use_momentum)
			sgd = new MomentumSGDTrainer(&model);
		else
			sgd = new SimpleSGDTrainer(&model);
		cerr << "%% Creating Encoder-Decoder ..."<<endl;
		EncoderDecoder<rnn_t> lm(model, LAYERS, EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_CHAR_DIM, INPUT_VOCAB_SIZE, INPUT_LEX_SIZE, &pretrained, singledir, nobase);
		cerr << "%%  Starting the training..." << endl;
		sup_train<rnn_t>(training, devel, &model,  &lm, report, WRITE_EVERY_I, sgd, fname, dc);
	}


	if (vm.count("decode")) {
		EncoderDecoder<rnn_t> lm(model, LAYERS, EMBEDDING_DIM, HIDDEN_DIM, EMBEDDING_CHAR_DIM, INPUT_VOCAB_SIZE, INPUT_LEX_SIZE,0, singledir, nobase);

		if (vm.count("initialise")) {
                	cerr << "initialising the model from: " << vm["initialise"].as<string>() << endl;
                	initialise(model, vm["initialise"].as<string>());
        	}
		else
			return 1;

		vector<VocabEntryPtr> test;
		cerr << "Reading test data from " << vm["test"].as<string>() << "...\n";
		if (vm.count("test")) {
			if (! exists(vm["test"].as<string>())) {
				cout << "the test file doesnt exist" << endl;
				return 1;
			}
			ReadFile(vm["test"].as<string>().c_str(), test);
		}

		run_decode<rnn_t>(test, &lm, dc);
	}

}




void initialise(Model &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	ifstream in(filename);
	boost::archive::text_iarchive ia(in);
	ia >> model;
}


// { a l i g n } ||| { a l i g n m e n t } ||| { a l i g n m e n t s } ||| <s> VIOLENT groups use rage and weapons to their advantage and sometimes terrorism PEACEFUL ||| depict pacifist and anti-war movements </s>

void ReadFile(const char* fname, vector<VocabEntryPtr>& dataset)
{
	std::string line, token;

	ifstream in(fname);
	if(!in.is_open()){
		cerr << "Failed to open file " << fname << endl;
		exit(1);
	}

	assert(in);

	unsigned tlc=0;
	while (getline(in, line)) {
		VocabEntryPtr entry(new VocabEntry());
		istringstream ss(line.c_str());
		int state = 0;
		while (ss >> token)
		{
			if (token == "|||"){
				state += 1;
				assert(state <= 6);
				continue;
			}


			if (state == 0)
				entry->Base.push_back(dc.convert(token));
			else if (state == 1)
				entry->Derived.push_back(dc.convert(token));
			else if (state == 2 || state == 3 || state==4)  // skipping the word form
				continue;
			else if (state == 5)
				entry->LeftContext.push_back(d.convert(token));
			else
				entry->RightContext.push_back(d.convert(token));
		}
			if (entry->LeftContext.front() != kSOS && entry->RightContext.back() != kEOS) {
				cerr << "The sentence in " << fname << ":" << tlc << " didn't start or end with <s>, </s>\n";
				abort();
			}
		++tlc;
		dataset.push_back(entry);
	}

	cerr << tlc << " lines, " <<  d.size() << " types, " << dc.size() << " chars\n" ;
}


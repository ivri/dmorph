#ifndef DERIV_H
#define DERIV_H

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>


using namespace std;
using namespace cnn;
using namespace cnn::expr;


typedef std::vector<std::string> Sentence;
typedef boost::shared_ptr<Sentence> SentencePtr;


typedef std::vector<int>               EncodedSentence;
typedef boost::shared_ptr<EncodedSentence>  EncodedSentencePtr;

///////////////////////////////////////////////////////////////
// VOCAB ENTRY 
///////////////////////////////////////////////////////////////
struct VocabEntry
{
	EncodedSentence LeftContext;
	EncodedSentence RightContext;
	std::vector<int> Derived;
	std::vector<int> Base; //char?
};

typedef boost::shared_ptr<VocabEntry> VocabEntryPtr;

cnn::Dict d, dc;
int kSOS,kEOS;

//parameters
unsigned LAYERS = 3;
unsigned INPUT_DIM = 200;
unsigned INPUT_CHAR_DIM = 15;
unsigned HIDDEN_DIM = 200;
unsigned INPUT_VOCAB_SIZE = 0;
unsigned INPUT_LEX_SIZE = 0;

int kSOW;
int kEOW;

template <class Builder>
struct EncoderDecoder {
	LookupParameter p_c;
	LookupParameter p_ec;  // map input to embedding (used in fwd and rev models)
	Parameter p_ie2h;
	Parameter p_bie;
	Parameter p_h2oe;
	Parameter p_boe;
	Parameter p_R;
	Parameter p_bias;
	Builder dec_builder;
	Builder left_rev_enc_builder;
	Builder left_fwd_enc_builder;
	Builder right_rev_enc_builder;
	Builder right_fwd_enc_builder; 
	Builder base_rev_enc_builder;
	Builder base_fwd_enc_builder; 
	explicit EncoderDecoder(Model& model) :
		dec_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model),
		left_rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		left_fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		right_rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		right_fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		base_rev_enc_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model),
		base_fwd_enc_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model){
			p_ie2h = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * 6)});
			p_bie = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)});
			p_h2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)});
			p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS)});
			p_c = model.add_lookup_parameters(INPUT_LEX_SIZE, {INPUT_CHAR_DIM}); 
			p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
			p_R = model.add_parameters({INPUT_LEX_SIZE, HIDDEN_DIM});
			p_bias = model.add_parameters({INPUT_LEX_SIZE});
		}

	// build graph and return Expression for total loss
	Expression BuildGraph(const vector<int>& leftsent, const vector<int>& rightsent,const vector<int>& base, const vector<int>& derived, ComputationGraph& cg) {

		// LEFT CONTEXT
		// forward encoder for left context
		left_fwd_enc_builder.new_graph(cg);
		left_fwd_enc_builder.start_new_sequence();
		for (unsigned t = 0; t < leftsent.size(); ++t) {
			Expression i_x_t = lookup(cg,p_ec,leftsent[t]);
			left_fwd_enc_builder.add_input(i_x_t);
		}
		// backward encoder for left context
		left_rev_enc_builder.new_graph(cg);
		left_rev_enc_builder.start_new_sequence();
		for (int t = leftsent.size() - 1; t >= 0; --t) {
			Expression i_x_t = lookup(cg, p_ec, leftsent[t]);
			left_rev_enc_builder.add_input(i_x_t);
		}

		// RIGHT CONTEXT
		// forward encoder for right context
		right_fwd_enc_builder.new_graph(cg);
		right_fwd_enc_builder.start_new_sequence();
		for (unsigned t = 0; t < rightsent.size(); ++t) {
			Expression i_x_t = lookup(cg,p_ec,rightsent[t]);
			right_fwd_enc_builder.add_input(i_x_t);
		}
		// backward encoder for left context
		right_rev_enc_builder.new_graph(cg);
		right_rev_enc_builder.start_new_sequence();
		for (int t = rightsent.size() - 1; t >= 0; --t) {
			Expression i_x_t = lookup(cg, p_ec, rightsent[t]);
			right_rev_enc_builder.add_input(i_x_t);
		}

		// CHAR-LEVEL BASE FORM 
		// forward encoder for right context
		base_fwd_enc_builder.new_graph(cg);
		base_fwd_enc_builder.start_new_sequence();
		for (unsigned t = 0; t < base.size(); ++t) {
			Expression i_x_t = lookup(cg,p_c,base[t]);
			base_fwd_enc_builder.add_input(i_x_t);
		}
		// backward encoder for left context
		base_rev_enc_builder.new_graph(cg);
		base_rev_enc_builder.start_new_sequence();
		for (int t = base.size() - 1; t >= 0; --t) {
			Expression i_x_t = lookup(cg, p_c, base[t]);
			base_rev_enc_builder.add_input(i_x_t);
		}	

		// encoder -> decoder transformation
		vector<Expression> to;
		for (auto h_l : right_fwd_enc_builder.final_h()) to.push_back(h_l);
		for (auto h_l : right_rev_enc_builder.final_h()) to.push_back(h_l);

		for (auto h_l : left_fwd_enc_builder.final_h()) to.push_back(h_l);
		for (auto h_l : left_rev_enc_builder.final_h()) to.push_back(h_l);

		for (auto h_l : base_fwd_enc_builder.final_h()) to.push_back(h_l);
		for (auto h_l : base_rev_enc_builder.final_h()) to.push_back(h_l);

		Expression i_combined = concatenate(to);
		Expression i_ie2h = parameter(cg, p_ie2h);
		Expression i_bie = parameter(cg, p_bie);
		Expression i_t = i_bie + i_ie2h * i_combined;
		cg.incremental_forward();
		Expression i_h = rectify(i_t); //replace with tanh?
		Expression i_h2oe = parameter(cg,p_h2oe);
		Expression i_boe = parameter(cg,p_boe);
		Expression i_nc = i_boe + i_h2oe * i_h;

		vector<Expression> oein1, oein2, oein;
		for (unsigned i = 0; i < LAYERS; ++i) {
			oein1.push_back(pickrange(i_nc, i * HIDDEN_DIM, (i + 1) * HIDDEN_DIM));
			oein2.push_back(tanh(oein1[i]));
		}
		for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein1[i]);
		for (unsigned i = 0; i < LAYERS; ++i) oein.push_back(oein2[i]);

		dec_builder.new_graph(cg);
		dec_builder.start_new_sequence(oein);

		// decoder
		Expression i_R = parameter(cg,p_R);
		Expression i_bias = parameter(cg,p_bias);
		vector<Expression> errs;

		const unsigned derlen = derived.size() - 1;
		for (unsigned t = 0; t < derlen; ++t) {
			Expression i_x_t = lookup(cg, p_c, derived[t]);
			Expression i_y_t = dec_builder.add_input(i_x_t);
			Expression i_r_t = i_bias + i_R * i_y_t;
			Expression i_ydist = log_softmax(i_r_t);
			errs.push_back(pick(i_ydist,derived[t+1]));
		}
		Expression i_nerr = sum(errs);
		return -i_nerr;
	}
};

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
				assert(state <= 4);
				continue;
			}


			if (state == 0)
				entry->Base.push_back(dc.convert(token));
			else if (state == 1)
				entry->Derived.push_back(dc.convert(token));
			else if (state == 2)   // skipping the word form
				continue;
			else if (state == 3)
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
void LoadData(const char*const* argv)
{
	kSOS = d.convert("<s>");
	kEOS = d.convert("</s>");
	kSOW = dc.convert("{");
	kEOW = dc.convert("}");
	vector<VocabEntryPtr> training, dev;

	Model model;
	cerr << "Reading training data from " << argv[1] << "...\n";
	ReadFile(argv[1], training);

	//	if (argv[2]){
	//	unsigned lid = 0;
	//       auto &lparams = model.lookup_parameters_list();
	//        for (const auto &p : model.lookup_parameters_list())  {
	//            for (unsigned i = 0; i < p->values.size(); ++i)
	//               memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(cnn::real) * p->values[i].d.size());
	//            lid++;
	//       	}
	//	}
	d.freeze();
	d.set_unk("UNK");
	dc.freeze();

	cerr << "Reading development data from " << argv[2] << "...\n";
	ReadFile(argv[2], dev);

	INPUT_VOCAB_SIZE = d.size();
	INPUT_LEX_SIZE = dc.size();
	ostringstream os;
	os << "bilm"
		<< '_' << LAYERS
		<< '_' << INPUT_DIM
		<< '_' << HIDDEN_DIM
		<< "-pid" << getpid() << ".params";
	const string fname = os.str();
	cerr << "Parameters will be written to: " << fname << endl;
	double best = 9e+99;

	bool use_momentum = false;
	Trainer* sgd = nullptr;
	if (use_momentum)
		sgd = new MomentumSGDTrainer(&model);
	else
		sgd = new SimpleSGDTrainer(&model);


	//RNNBuilder rnn(LAYERS, INPUT_DIM, HIDDEN_DIM, &model);
	//EncoderDecoder<SimpleRNNBuilder> lm(model);
	EncoderDecoder<LSTMBuilder> lm(model);
	//	if (argc == 4) {
	//		string fname = argv[3];
	//		ifstream in(fname);
	//		boost::archive::text_iarchive ia(in);
	//		ia >> model;
	//	}


	unsigned report_every_i = 50;// 50!
	unsigned dev_every_i_reports = 10;
	unsigned si = training.size();
	vector<unsigned> order(training.size());
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
	bool first = true;
	int report = 0;
	unsigned lines = 0;
	while(1) {
		Timer iteration("completed in");
		double loss = 0;
		unsigned chars = 0;
		for (unsigned i = 0; i < report_every_i; ++i) {
			if (si == training.size()) {
				si = 0;
				if (first) { first = false; } else { sgd->update_epoch(); }
				cerr << "**SHUFFLE\n";
				random_shuffle(order.begin(), order.end());
			}

			// build graph for this instance
			ComputationGraph cg;
			auto entry = training[order[si]];
			chars += entry->Derived.size() - 1;
			++si;
			lm.BuildGraph(entry->LeftContext, entry->RightContext, entry->Base, entry->Derived, cg);
			loss += as_scalar(cg.forward());
			cg.backward();
			sgd->update();
			++lines;
		}
		sgd->status();
		cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';


#if 0
		lm.RandomSample();
#endif

		// show score on dev data?
		report++;
		if (report % dev_every_i_reports == 0) {
			double dloss = 0;
			int dchars = 0;
			for (auto entry : dev) {
				ComputationGraph cg;
				lm.BuildGraph(entry->LeftContext, entry->RightContext, entry->Base, entry->Derived, cg);
				dloss += as_scalar(cg.forward());
				dchars += entry->Derived.size() - 1;
			}
			if (dloss < best) {
				best = dloss;
				ofstream out(fname);
				boost::archive::text_oarchive oa(out);
				oa << model;
			}
			cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
		}
	}
}

#endif

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


template <class Builder>
struct EncoderDecoder {
	LookupParameter p_c;
	//	LookupParameter p_t; // pretrained word embeddings 
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
	unsigned layers;
	unsigned embedding_dim;
	unsigned hidden_dim;
	unsigned num_init_params;
	bool single_dir = false;
	bool nobase = false;

	bool  hasNan (const vector<float> &a) {
		for (size_t i = 0; i < a.size(); ++i) if (isnan(a[i])) return true;
		return false;
	}


	bool  hasNan (Expression* x) {
		vector<float>a (as_vector(x->value() ));
		for (size_t i = 0; i < a.size(); ++i) if (isnan(a[i])) return true;
		return false;
	}

	explicit EncoderDecoder(Model& model, unsigned LAYERS, unsigned INPUT_DIM, unsigned HIDDEN_DIM, unsigned INPUT_CHAR_DIM, unsigned INPUT_VOCAB_SIZE, unsigned INPUT_LEX_SIZE, const unordered_map<unsigned, vector<float>>* pretrained=0, bool _single_dir=false, bool _nobase=false) :
		dec_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model),
		left_rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		left_fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		right_rev_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		right_fwd_enc_builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model),
		base_rev_enc_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model),
		base_fwd_enc_builder(LAYERS, INPUT_CHAR_DIM, HIDDEN_DIM, &model){
			p_bie = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5)});
			p_h2oe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS), unsigned(HIDDEN_DIM * LAYERS * 1.5)});
			p_boe = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS)});
			p_c = model.add_lookup_parameters(INPUT_LEX_SIZE, {INPUT_CHAR_DIM}); 
			p_ec = model.add_lookup_parameters(INPUT_VOCAB_SIZE, {INPUT_DIM}); 
			p_R = model.add_parameters({INPUT_LEX_SIZE, HIDDEN_DIM});
			p_bias = model.add_parameters({INPUT_LEX_SIZE});

			int concat_len = 6; //left*2+righ+2+base*2 	
			this->layers = LAYERS;
			this->hidden_dim = HIDDEN_DIM;
			if (pretrained) {
				cerr << "%% Initializing the vectors..." << endl;
				for (const auto& it : *pretrained) {
					p_ec.initialize(it.first, it.second);
				}
			} 
			if (_single_dir) {
				cerr << "%% The model will be single-directed, i.e. no bidirectionality for contexts." << endl;
				this->single_dir=true;
				concat_len -= 2;
			}

			if (_nobase) {
				cerr << "%% No base form is included. The model entirely relies on the context." << endl;
				this->nobase=true;
				concat_len -= 2;
			}
			p_ie2h = model.add_parameters({unsigned(HIDDEN_DIM * LAYERS * 1.5), unsigned(HIDDEN_DIM * LAYERS * concat_len)}); //6
		}

	// build graph and return Expression for total loss
	vector<Expression> BuildGraph(const vector<int>& leftsent, const vector<int>& rightsent,const vector<int>& base, ComputationGraph& cg) {

		// LEFT CONTEXT
		// forward encoder for left context

		left_fwd_enc_builder.new_graph(cg);
		left_fwd_enc_builder.start_new_sequence();
		for (unsigned t = 0; t < leftsent.size(); ++t) {
			Expression i_x_t = lookup(cg,p_ec,leftsent[t]);
			if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-1!" << endl, abort();
			left_fwd_enc_builder.add_input(i_x_t);
		}
		if (!single_dir) {
//			cerr << "Adding 2nd left direction..." << endl;
			// backward encoder for left context
			left_rev_enc_builder.new_graph(cg);
			left_rev_enc_builder.start_new_sequence();
			for (int t = leftsent.size() - 1; t >= 0; --t) {
				Expression i_x_t = lookup(cg, p_ec, leftsent[t]);
				if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-2!" << endl, abort();
				left_rev_enc_builder.add_input(i_x_t);
			}
		}

		// RIGHT CONTEXT
		// forward encoder for right context
		if (!single_dir) {
//			cerr << "Adding 2nd right direction..." << endl;
			right_fwd_enc_builder.new_graph(cg);
			right_fwd_enc_builder.start_new_sequence();
			for (unsigned t = 0; t < rightsent.size(); ++t) {
				Expression i_x_t = lookup(cg,p_ec,rightsent[t]);
				if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-3!" << endl, abort();
				right_fwd_enc_builder.add_input(i_x_t);
			}
		}
		// backward encoder for left context
		right_rev_enc_builder.new_graph(cg);
		right_rev_enc_builder.start_new_sequence();
		for (int t = rightsent.size() - 1; t >= 0; --t) {
			Expression i_x_t = lookup(cg, p_ec, rightsent[t]);
			if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-4!" << endl, abort();
			right_rev_enc_builder.add_input(i_x_t);
		}

		// CHAR-LEVEL BASE FORM 
		// forward encoder for right context
		if (!nobase) {
//			cerr << "Adding base both directions..." << endl;
			base_fwd_enc_builder.new_graph(cg);
			base_fwd_enc_builder.start_new_sequence();
			for (unsigned t = 0; t < base.size(); ++t) {
				Expression i_x_t = lookup(cg,p_c,base[t]);
				if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-5!" << endl, abort();
				base_fwd_enc_builder.add_input(i_x_t);
			}
			// backward encoder for left context
			base_rev_enc_builder.new_graph(cg);
			base_rev_enc_builder.start_new_sequence();
			for (int t = base.size() - 1; t >= 0; --t) {
				Expression i_x_t = lookup(cg, p_c, base[t]);
				if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 0-6!" << endl, abort();
				base_rev_enc_builder.add_input(i_x_t);
			}	
		}

		// encoder -> decoder transformation
		vector<Expression> to;
		if (!single_dir)
			for (auto h_l : right_fwd_enc_builder.final_h()) to.push_back(h_l);
		for (auto h_l : right_rev_enc_builder.final_h()) to.push_back(h_l);

		for (auto h_l : left_fwd_enc_builder.final_h()) to.push_back(h_l);
		if (!single_dir)
			for (auto h_l : left_rev_enc_builder.final_h()) to.push_back(h_l);
		if (!nobase) {
			for (auto h_l : base_fwd_enc_builder.final_h()) to.push_back(h_l);
			for (auto h_l : base_rev_enc_builder.final_h()) to.push_back(h_l);
		}
		Expression i_combined = concatenate(to);
		if (hasNan(&i_combined)) cerr << "Achtung ! NAN 0-7!" << endl, abort();
		Expression i_ie2h = parameter(cg, p_ie2h);
		Expression i_bie = parameter(cg, p_bie);
		Expression i_t = i_bie + i_ie2h * i_combined;
		if (hasNan(&i_t)) cerr << "Achtung ! NAN 0-8!" << endl, abort();
		cg.incremental_forward();
		Expression i_h = rectify(i_t); //replace with tanh?
		if (hasNan(&i_h)) cerr << "Achtung ! NAN 0-9!" << endl, abort();
		Expression i_h2oe = parameter(cg,p_h2oe);
		if (hasNan(&i_h2oe)) cerr << "Achtung ! NAN 0-10!" << endl, abort();
		Expression i_boe = parameter(cg,p_boe);
		if (hasNan(&i_boe)) cerr << "Achtung ! NAN 0-11!" << endl, abort();
		Expression i_nc = i_boe + i_h2oe * i_h;
		if (hasNan(&i_nc)) cerr << "Achtung ! NAN 0-12!" << endl, abort();

		vector<Expression> oein1, oein2, oein;
		for (unsigned i = 0; i < layers; ++i) {
			oein1.push_back(pickrange(i_nc, i * hidden_dim, (i + 1) * hidden_dim));
			oein2.push_back(tanh(oein1[i]));
		}
		for (unsigned i = 0; i < layers; ++i) oein.push_back(oein1[i]);
		for (unsigned i = 0; i < layers; ++i) oein.push_back(oein2[i]);

		return oein;
	}

	Expression Propagate(const vector<int>& leftsent, const vector<int>& rightsent, const vector<int>& base, const vector<int>& derived, ComputationGraph& cg, const cnn::Dict &dc) {

		vector<Expression> oein = BuildGraph(leftsent, rightsent, base, cg);
		dec_builder.new_graph(cg);
		dec_builder.start_new_sequence(oein);

		// decoder
		Expression i_R = parameter(cg,p_R);
		if (hasNan(&i_R)) cerr << "Achtung ! NAN 1-0!" << endl, abort();
		Expression i_bias = parameter(cg,p_bias);
		if (hasNan(&i_bias)) cerr << "Achtung ! NAN 1-2!" << endl, abort();
		vector<Expression> errs;
		cerr << endl;
		const unsigned derlen = derived.size() - 1;
		for (unsigned t = 0; t < derlen; ++t) {
			Expression i_x_t = lookup(cg, p_c, derived[t]);
			if (hasNan(&i_x_t)) cerr << "Achtung ! NAN 1-3 !" << endl, abort();
			cerr << dc.convert(derived[t+1]) << "->";
			Expression i_y_t = dec_builder.add_input(i_x_t);
			if (hasNan(&i_y_t)) cerr << "Achtung ! NAN 1-4!" << endl, abort();
			Expression i_r_t = i_bias + i_R * i_y_t;
			if (hasNan(&i_r_t)) cerr << "Achtung ! NAN 1-5 !" << endl, abort();
			Expression i_ydist = log_softmax(i_r_t);
			//	cerr << "i_r_t" << cg.get_value(i_r_t) <<"  i_ydist = " << cg.get_value(i_ydist)<< endl;
			cg.incremental_forward();
			auto dist = as_vector(cg.get_value(i_ydist));
			//for (auto item = dist.begin(); item != dist.end(); ++item)
			//  std::cout << *item << ' ';
			auto next = std::max_element(dist.begin(), dist.end()) - dist.begin();
			cerr << dc.convert(next)  << " ";
			//errs.push_back(pickneglogsoftmax(i_r_t, derived[t+1]));
			errs.push_back(pick(i_ydist,derived[t+1]));//i_ydist
			//			cerr << " " << dc.convert(derived[t+1]) << " Error= "<<as_scalar(cg.get_value(errs[t]));
		}
		//		cerr << "Fin!" << endl;
		Expression i_nerr = sum(errs);
		//		cg.incremental_forward();
		cerr <<"Err = " << cg.get_value(i_nerr)<< endl;
		return -i_nerr;
	}


	// build graph and return Expression for total loss
	vector<int> Decode(const vector<int>& leftsent, const vector<int>& rightsent, const vector<int>& base, ComputationGraph& cg) {

		vector<Expression> oein = BuildGraph(leftsent, rightsent, base, cg);
		dec_builder.new_graph(cg);
		dec_builder.start_new_sequence(oein);
		int next = 0;//kSOW;
		// decoder
		Expression i_R = parameter(cg,p_R);
		Expression i_bias = parameter(cg,p_bias);
		vector<int> result;
		//int t = 0;
		//		cout << "=====================" << endl;
		do {
			Expression i_x_t = lookup(cg, p_c, next);
			Expression i_y_t = dec_builder.add_input(i_x_t);
			Expression i_r_t = i_bias + i_R * i_y_t;
			Expression i_ydist = log_softmax(i_r_t);
			auto dist = as_vector(cg.incremental_forward());
			next = std::max_element(dist.begin(), dist.end()) - dist.begin();
			//			for (auto item = dist.begin(); item != dist.end(); ++item)
			//                                std::cout << *item << ' ';
			//			cout << "\nnext = "<< dc.convert(next) << "  i=" << i << endl;
			result.push_back(next);
			//	errs.push_back(pick(dist,derived[t+1]));
			//	++i;
		} 
		while (next != 1);//kEOW );//&& i != 20); //EOW

		return result;
	}

	private:
	EncoderDecoder(const EncoderDecoder &) { cerr << "Copying Enc-Dec instance" << endl; exit(1);}
};
/*
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

 */
	template <class Builder>
void sup_train(const vector<VocabEntryPtr> &training, const vector<VocabEntryPtr> &devel,Model *model,
		EncoderDecoder<Builder> *lm, unsigned report_every_i, unsigned dev_every_i_reports, Trainer* sgd,
		const string &fname, const cnn::Dict &dc)
{
	double best = 9e+99;

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
				cerr << "\nUpdating the epoch...\n";
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
			cerr << "\nsi = "<< si << " Derived: ";
			for (std::vector<int>::const_iterator item = entry->Derived.begin(); item != entry->Derived.end(); ++item)
				cerr << dc.convert(*item) << ' ';
			cerr << " Base: ";
			for (std::vector<int>::const_iterator item = entry->Base.begin(); item != entry->Base.end(); ++item)
				cerr << dc.convert(*item) << ' ';
			//			cerr << " Left: ";
			//			for (std::vector<int>::const_iterator item = entry->LeftContext.begin(); item != entry->LeftContext.end(); ++item)
			//                        	cerr << *item << ' ';
			//			cerr << " Right: ";
			//			for (std::vector<int>::const_iterator item = entry->RightContext.begin(); item != entry->RightContext.end(); ++item)
			//                                cerr << *item << ' ';
			cerr << endl;
			lm->Propagate(entry->LeftContext, entry->RightContext, entry->Base, entry->Derived, cg, dc);
			loss += as_scalar(cg.forward());
			cg.backward();
			sgd->update();
			++lines;
		}
		sgd->status();
		cerr << "Loss = " << loss << " chars = " << chars << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';


#if 0
		lm.RandomSample();
#endif

		// show score on dev data?
		report++;
		if (report % dev_every_i_reports == 0) {
			cerr <<"Evaluating development\n";
			double dloss = 0;
			int dchars = 0;
			for (auto entry : devel) {
				ComputationGraph cg;
				lm->Propagate(entry->LeftContext, entry->RightContext, entry->Base, entry->Derived, cg, dc);
				//					for (std::vector<int>::const_iterator item = entry->Base.begin(); item != entry->Base.end(); ++item)
				//		                                std::cout << dc.convert(*item) << ',';

				dloss += as_scalar(cg.forward());
				dchars += entry->Derived.size() - 1;
			}
			if (dloss < best) {
				best = dloss;
				ofstream out(fname);
				cerr << "Saving the model into the file...." << endl;
				boost::archive::text_oarchive oa(out);
				oa << *model;
			}
			cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] Loss = " << dloss << " Chars = " << dchars  <<" E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
		}
	}
}
	template <class Builder>
void run_decode(const vector<VocabEntryPtr> &test, EncoderDecoder<Builder> *lm,const cnn::Dict &dc)
{
	double loss = 0;
	double chars = 0;
	for (auto entry : test) {
		cout << "\n";
		ComputationGraph cg;
		for (std::vector<int>::const_iterator item = entry->Derived.begin(); item != entry->Derived.end(); ++item)
			std::cout << dc.convert(*item) << ' ';//dc.convert(*item) << ' ';
		std::cout << "|||" ;
		// loss += as_scalar(cg.forward());
		// chars += entry->Derived.size() - 1;
		std::vector<int> res = lm->Decode(entry->LeftContext, entry->RightContext, entry->Base, cg);
		cg.forward();
		//chars += entry->Derived.size() - 1;
		for (std::vector<int>::const_iterator item = res.begin(); item != res.end(); ++item)
			std::cout << dc.convert(*item) << ' ';//dc.convert(*item) << ' '; FIXX
		//cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
	}
}

#endif

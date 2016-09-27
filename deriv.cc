#include "deriv.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>



int main(int argc, char** argv) {
	cnn::initialize(argc, argv);
	if (argc != 3 && argc != 4) {
		cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
		return 1;
	}
	LoadData(argv);
}

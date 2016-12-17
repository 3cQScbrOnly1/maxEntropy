#include "Utiltensor.h"

#include <map>

using namespace std;

class WordEmbReader
{
	public:
		void readWordEmb(NRMat<double> &word_emb,map<string,int> features_alphabet,char *word_emb_path);
		void delNull(vector<string> &data);
};

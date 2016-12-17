#include "WordEmbReader.h"

#include <fstream>
#include <boost/algorithm/string.hpp>
#include <sstream>

using namespace std;
using namespace boost;

void WordEmbReader::readWordEmb(NRMat<double> &word_emb,map<string,int> features_alphabet,char *word_emb_path)
{
	ifstream w_e(word_emb_path);	
	if(w_e.is_open())
	{
		string line;
		if(getline(w_e,line))
		{
			vector<string> data;
			split(data,line,is_any_of(" "));
			delNull(data);
			map<string,int>::iterator it = features_alphabet.find(data[0]);
			word_emb.resize(features_alphabet.size(),data.size()-1);
			word_emb = 0.0;
			if(it != features_alphabet.end())
			{
				for(int i = 1; i < data.size(); i++)
				{
					stringstream string_to_double;
					double cur_value;
					string_to_double << data[i];
					string_to_double >> cur_value;
					word_emb[it -> second][i] = cur_value;
				}
			}
		}
		while(getline(w_e,line))
		{
			vector<string> data;	
			split(data,line,is_any_of(" "));
			delNull(data);
			map<string,int>::iterator it = features_alphabet.find(data[0]);
			if(it != features_alphabet.end())
			{
				for(int i = 1; i < data.size(); i++)
				{
					stringstream string_to_double;
					double cur_value;
					string_to_double << data[i];
					string_to_double >> cur_value;
					word_emb[it -> second][i] = cur_value;
				}
			}
		}
	} else
		cout << "open word embedding error." << endl;
	w_e.close();
}

void WordEmbReader::delNull(vector<string> &data)
{
	vector<string>::iterator it = data.begin(); 
	while(it != data.end())
	{
		if(*it == "")
			data.erase(it);
		else
			it++;
	}
}

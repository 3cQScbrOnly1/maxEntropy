#include "Example.h"
#include "MyAlphabet.h"
#include "Instance.h"

class InstanceToExampleTransformer
{
	private:
		void labelTransfer(vector<string> labels_alphabet,string label,vector<int> &d_label);
		void featuresTransfer(map<string,int> features_alphabet,vector<string> features,vector<int> &d_features);
	public:
		void transfer(MyAlphabet the_alphabet,vector<Instance> the_instances,vector<Example> &the_examples);
};

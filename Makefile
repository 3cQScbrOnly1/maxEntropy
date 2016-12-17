vpath %.h mshadow %.h LibN3L
OBJ = main.o InstanceReader.o MyAlphabet.o FeatureExtractor.o InstanceToExampleTransformer.o Classifier.o SparseLayer.o Counter.o WClassifier.o UniLayer.o WordEmbReader.o

CUDA=-DMSHADOW_USE_CUDA=0 
CBLAS=-DMSHADOW_USE_CBLAS=1 
MKL=-DMSHADOW_USE_MKL=0 

DEPEND=$(CUDA)$(CBLAS)$(MKL)


myME:$(OBJ)
	g++  -lopenblas -o myME $(OBJ)
main.o:InstanceReader.o InstanceToExampleTransformer.o MyAlphabet.o Classifier.o FeatureExtractor.o SparseLayer.o Counter.o WClassifier.o UniLayer.o main.cpp
	g++ -c $(DEPEND) main.cpp -Imshadow -ILibN3L -Wno-deprecated
InstanceReader.o:InstanceReader.cpp InstanceReader.h Instance.h
	g++ -c InstanceReader.cpp
MyAlphabet.o:MyAlphabet.cpp MyAlphabet.h FeatureExtractor.o
	g++ -c MyAlphabet.cpp
FeatureExtractor.o:FeatureExtractor.h FeatureExtractor.cpp
	g++ -c FeatureExtractor.cpp
InstanceToExampleTransformer.o:MyAlphabet.o Instance.h Example.h  InstanceToExampleTransformer.cpp InstanceToExampleTransformer.h
	g++ -c InstanceToExampleTransformer.cpp
Classifier.o:Classifier.cpp Classifier.h SparseLayer.o Counter.o Example.h
	g++ -c $(DEPEND) Classifier.cpp -Imshadow
SparseLayer.o:SparseLayer.cpp SparseLayer.h
	g++ -c $(DEPEND) SparseLayer.cpp -Imshadow -ILibN3L -Wno-deprecated
WClassifier.o:WClassifier.cpp WClassifier.h UniLayer.o Counter.o Example.h
	g++ -c $(DEPEND) WClassifier.cpp -Imshadow -ILibN3L -Wno-deprecated
UniLayer.o:UniLayer.cpp UniLayer.h
	g++ -c $(DEPEND) UniLayer.cpp -Imshadow -ILibN3L -Wno-deprecated
Counter.o::Counter.cpp Counter.h
	g++ -c Counter.cpp
WordEmbReader.o:WordEmbReader.cpp WordEmbReader.h
	g++ -c $(DEPEND) WordEmbReader.cpp -ILibN3L -Imshadow -Wno-deprecated

clean:
	rm myME $(OBJ)

#include "NN.h"
#include <math.h>


NN::NN()
{
}


NN::~NN()
{
}

/**
* ���͑w�ւ̓��͂�ݒ肷��
* @param i �m�[�h�ԍ�
* @param value �l
*/
void setInput(int i, double value) {
	if ((i >= 0) && (i < inputLayer.numNodes)) {
		inputLayer.neuronValues[i] = value;
	}
}

/**
* �o�͑w�̏o�͂𓾂�
* @param i �m�[�h�ԍ�
* @return �o�͑wi�Ԗڂ̃m�[�h�̏o�͒l
*/
double getOutput(int i) {
	if ((i >= 0) && (i < outputLayer.numNodes)) {
		return outputLayer.neuronValues[i];
	}

	return Double.MAX_VALUE;  // �G���[
}

/**
* ���t�M����ݒ肷��
* @param i �m�[�h�ԍ�
* @param value ���t�M���̒l
*/
void setTeacherValue(int i, double value) {
	if ((i >= 0) && (i < outputLayer.numNodes)) {
		outputLayer.teacherValues[i] = value;
	}
}

/**
* �O�����`�d�i���Ԃ͏d�v�j
*/
void feedForward() {
	inputLayer.calculateNeuronValues();
	hiddenLayer.calculateNeuronValues();
	outputLayer.calculateNeuronValues();
}

/**
* �t�����`�d�i���Ԃ͏d�v�j
*/
void backPropagate() {
	outputLayer.calculateErrors();
	hiddenLayer.calculateErrors();

	hiddenLayer.adjustWeights();
	inputLayer.adjustWeights();
}

/**
* �o�͑w�ōő�o�͂����m�[�h�ԍ���Ԃ�
* @return �ő�o�͂����m�[�h�ԍ�
*/
int getMaxOutputID() {
	double max = outputLayer.neuronValues[0];
	int id = 0;

	for (int i = 1; i<outputLayer.numNodes; i++) {
		if (outputLayer.neuronValues[i] > max) {
			max = outputLayer.neuronValues[i];
			id = i;
		}
	}

	return id;
}

/**
* �o�͂Ƌ��t�M���̕���2��덷���v�Z����
* @return ����2��덷
*/
double calculateError() {
	double error = 0;

	for (int i = 0; i<outputLayer.numNodes; i++) {
		error += pow(outputLayer.neuronValues[i] - outputLayer.teacherValues[i], 2);
	}

	error /= outputLayer.numNodes;

	return error;
}

/**
* �w�K����ݒ肷��
* @param rate �w�K��
*/
void setLearningRate(double rate) {
	inputLayer.learningRate = rate;
	hiddenLayer.learningRate = rate;
	outputLayer.learningRate = rate;
}
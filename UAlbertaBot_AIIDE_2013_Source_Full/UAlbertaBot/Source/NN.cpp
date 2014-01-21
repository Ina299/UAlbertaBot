#include "NN.h"
#include <math.h>


NN::NN()
{
}


NN::~NN()
{
}

/**
* 入力層への入力を設定する
* @param i ノード番号
* @param value 値
*/
void setInput(int i, double value) {
	if ((i >= 0) && (i < inputLayer.numNodes)) {
		inputLayer.neuronValues[i] = value;
	}
}

/**
* 出力層の出力を得る
* @param i ノード番号
* @return 出力層i番目のノードの出力値
*/
double getOutput(int i) {
	if ((i >= 0) && (i < outputLayer.numNodes)) {
		return outputLayer.neuronValues[i];
	}

	return Double.MAX_VALUE;  // エラー
}

/**
* 教師信号を設定する
* @param i ノード番号
* @param value 教師信号の値
*/
void setTeacherValue(int i, double value) {
	if ((i >= 0) && (i < outputLayer.numNodes)) {
		outputLayer.teacherValues[i] = value;
	}
}

/**
* 前向き伝播（順番は重要）
*/
void feedForward() {
	inputLayer.calculateNeuronValues();
	hiddenLayer.calculateNeuronValues();
	outputLayer.calculateNeuronValues();
}

/**
* 逆向き伝播（順番は重要）
*/
void backPropagate() {
	outputLayer.calculateErrors();
	hiddenLayer.calculateErrors();

	hiddenLayer.adjustWeights();
	inputLayer.adjustWeights();
}

/**
* 出力層で最大出力を持つノード番号を返す
* @return 最大出力を持つノード番号
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
* 出力と教師信号の平均2乗誤差を計算する
* @return 平均2乗誤差
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
* 学習率を設定する
* @param rate 学習率
*/
void setLearningRate(double rate) {
	inputLayer.learningRate = rate;
	hiddenLayer.learningRate = rate;
	outputLayer.learningRate = rate;
}
#pragma once
class NN
{
public:
	NN();
	~NN();
};

public class NeuralNetwork {
	private Layer inputLayer;   // 入力層
	private Layer hiddenLayer;  // 隠れ層
	private Layer outputLayer;  // 出力層

	/**
	* ニューラルネットを初期化
	* @param numInputNodes 入力層のノード数
	* @param numHiddenNodes 隠れ層のノード数
	* @param numOutputNodes 出力層のノード数
	*/
	public void init(int numInputNodes, int numHiddenNodes, int numOutputNodes) {
		inputLayer = new Layer();
		hiddenLayer = new Layer();
		outputLayer = new Layer();

		// 入力層（親層がないことに注意）
		inputLayer.numNodes = numInputNodes;
		inputLayer.numChildNodes = numHiddenNodes;
		inputLayer.numParentNodes = 0;
		inputLayer.init(numInputNodes, null, hiddenLayer);
		inputLayer.randomizeWeights();

		// 隠れ層
		hiddenLayer.numNodes = numHiddenNodes;
		hiddenLayer.numChildNodes = numOutputNodes;
		hiddenLayer.numParentNodes = numInputNodes;
		hiddenLayer.init(numHiddenNodes, inputLayer, outputLayer);
		hiddenLayer.randomizeWeights();

		// 出力層（子層がないことに注意）
		outputLayer.numNodes = numOutputNodes;
		outputLayer.numChildNodes = 0;
		outputLayer.numParentNodes = numHiddenNodes;
		outputLayer.init(numOutputNodes, hiddenLayer, null);
		// 出力層に重みはないのでrandomizeWeights()は必要ない
	}


}

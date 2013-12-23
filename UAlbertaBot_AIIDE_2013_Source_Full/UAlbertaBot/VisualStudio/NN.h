#pragma once
class NN
{
public:
	NN();
	~NN();
};

public class NeuralNetwork {
	private Layer inputLayer;   // “ü—Í‘w
	private Layer hiddenLayer;  // ‰B‚ê‘w
	private Layer outputLayer;  // o—Í‘w

	/**
	* ƒjƒ…[ƒ‰ƒ‹ƒlƒbƒg‚ğ‰Šú‰»
	* @param numInputNodes “ü—Í‘w‚Ìƒm[ƒh”
	* @param numHiddenNodes ‰B‚ê‘w‚Ìƒm[ƒh”
	* @param numOutputNodes o—Í‘w‚Ìƒm[ƒh”
	*/
	public void init(int numInputNodes, int numHiddenNodes, int numOutputNodes) {
		inputLayer = new Layer();
		hiddenLayer = new Layer();
		outputLayer = new Layer();

		// “ü—Í‘wie‘w‚ª‚È‚¢‚±‚Æ‚É’ˆÓj
		inputLayer.numNodes = numInputNodes;
		inputLayer.numChildNodes = numHiddenNodes;
		inputLayer.numParentNodes = 0;
		inputLayer.init(numInputNodes, null, hiddenLayer);
		inputLayer.randomizeWeights();

		// ‰B‚ê‘w
		hiddenLayer.numNodes = numHiddenNodes;
		hiddenLayer.numChildNodes = numOutputNodes;
		hiddenLayer.numParentNodes = numInputNodes;
		hiddenLayer.init(numHiddenNodes, inputLayer, outputLayer);
		hiddenLayer.randomizeWeights();

		// o—Í‘wiq‘w‚ª‚È‚¢‚±‚Æ‚É’ˆÓj
		outputLayer.numNodes = numOutputNodes;
		outputLayer.numChildNodes = 0;
		outputLayer.numParentNodes = numHiddenNodes;
		outputLayer.init(numOutputNodes, hiddenLayer, null);
		// o—Í‘w‚Éd‚İ‚Í‚È‚¢‚Ì‚ÅrandomizeWeights()‚Í•K—v‚È‚¢
	}


}

#pragma once

#include "Common.h"
#include "BWTA.h"
#include "base/BuildOrderQueue.h"
#include "InformationManager.h"
#include "base/WorkerManager.h"
#include "base/StarcraftBuildOrderSearchManager.h"
#include <sys/stat.h>
#include <cstdlib>
#include "floatfann.h"
#include "fann_cpp.h"
#include <ios>
#include <iostream>
#include <iomanip>
#include <boost/foreach.hpp>
#include <boost/tuple/tuple.hpp>
#include <math.h>
#include <time.h>
#include <map>

#include "..\..\StarcraftBuildOrderSearch\Source\starcraftsearch\StarcraftData.hpp"

//#include <bitset>

typedef std::pair<int, int> IntPair;
typedef std::pair<MetaType, UnitCountType> MetaPair;
typedef std::vector<MetaPair> MetaPairVector;

class Neural
{
	Neural();
	~Neural() {}

	std::string					readDir;
	std::string					writeDir;

	BWAPI::Race					selfRace;
	BWAPI::Race					enemyRace;

	std::vector<float> 	bestinput;

	std::vector<std::vector<float> >	inputs;
	std::vector<std::vector<float> >	outputs;

	std::vector<float>	states;

	static FANN::neural_net net;

	int count;

	const int num_actions=5;
	int num_states;
	const int unit_count = 2;
	//�������狭���w�K�̃p�����[�^
	const float gamma = 0.95;
	const float alpha = 0.7;

	//��������j���[�����l�b�g�̃p�����[�^
	const float learning_rate = 0.7f;
	const unsigned int num_layers = 3;
	unsigned int num_input;
	const unsigned int num_hidden = 3;
	const unsigned int num_output = 1;
	const float desired_error = 0.001f;

	const unsigned int max_iterations = 300000;

	const unsigned int iterations_between_reports = 1000;

//	void 	(FANN_API *createTrainDataset)(unsigned int, unsigned int, unsigned int, fann_type *, fann_type *);

	FANN::neural_net net;

	const	int					getScore(BWAPI::Player * player) const;

	void	setActions();

	void	selectBestAction();

	int print_callback(FANN::neural_net &net, FANN::training_data &train,
		unsigned int max_epochs, unsigned int epochs_between_reports,
		float desired_error, unsigned int epochs, void *user_data);

	void	strategy_test();

	void	createNetwork();

	bool	neuralUpdateFrame();

	void	setStates();
	//std::set<BWAPI::Unit *> & unitsToAssign
	std::vector<float>	& getState();

	int		setNumState();

public:

	static	Neural &	Instance();

	void				onEnd(const bool);

	void				update();

	const std::vector<float> &		getActions();


};

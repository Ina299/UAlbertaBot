#pragma once

#include "Common.h"
#include "BWTA.h"
#include "BWAPI/Unit.h"
#include "BWAPI/Region.h"
#include "base/BuildOrderQueue.h"
#include "InformationManager.h"
#include "base/WorkerManager.h"
#include "base/StarcraftBuildOrderSearchManager.h"
#include "floatfann.h"
#include "fann_cpp.h"
#include <sys/stat.h>
#include <cstdlib>
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
	~Neural(){}

	//std::string					readDir;
	//std::string					writeDir;

	BWAPI::Race					selfRace;
	BWAPI::Race					enemyRace;

//	std::vector<float> 	bestinput;

	std::vector<std::vector<float> >	inputs;
	std::vector<std::vector<float> >	outputs;

	std::vector<float>	states;

	int num_input;
	int num_states;

//	void 	(FANN_API *createTrainDataset)(unsigned int, unsigned int, unsigned int, fann_type *, fann_type *);

	const	int					getScore(BWAPI::Player * player) const;

	void	setActions();

	FANN::neural_net net;
	
	void	selectBestAction();

	/*
	int print_callback(FANN::neural_net &net, FANN::training_data &train,
		unsigned int max_epochs, unsigned int epochs_between_reports,
		float desired_error, unsigned int epochs, void *user_data);
		*/

//	void	strategy_test();

	void	createNetwork();

	bool	neuralUpdateFrame();

	void	setStates();
	//std::set<BWAPI::Unit *> & unitsToAssign
//	std::vector<float>	& getState();

	int		setNumState();

public:
	

	static	Neural &	Instance();

	void				onEnd(const bool);

	void				update();

	const std::vector<float> &		getActions();


};

#pragma once

#include "Common.h"
#include "BWTA.h"
#include <sys/stat.h>
#include <cstdlib>

#include "..\..\StarcraftBuildOrderSearch\Source\starcraftsearch\StarcraftData.hpp"

typedef std::pair<int, int> IntPair;
typedef std::pair<MetaType, UnitCountType> MetaPair;
typedef std::vector<MetaPair> MetaPairVector;

class Neural
{
	Neural();
	~Neural() {}

	std::string					readNetwork;
	std::string					writeNetwork;

	BWAPI::Race					selfRace;
	BWAPI::Race					enemyRace;

	bool						firstAttackSent;

	void	addStrategies();
	void	setStrategy();
	void	readResults();
	void	writeResults();

	const	int					getScore(BWAPI::Player * player) const;
	const	double				getUCBValue(const size_t & strategy) const;

public:

	static	Neural &	Instance();

	void				onEnd();

	std::vector<int> &		getActions();

	const	bool		regroup(int numInRadius);
	const	bool		doAttack(const std::set<BWAPI::Unit *> & freeUnits);
	const	int			defendWithWorkers();
	const	bool		rushDetected();


	const	int					getCurrentStrategy();

	const	MetaPairVector		getBuildOrderGoal();
	const	std::string			getOpeningBook() const;
};

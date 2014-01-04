#include "Common.h"
#include "Neural.h"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::left;
using std::right;
using std::showpos;
using std::noshowpos;


// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
,enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	createNetwork();
	setActions();
	addActions();
}

// get an instance of this
Neural & Neural::Instance()
{
	static Neural instance;
	return instance;
}

// read weight of Network

void Neural::readNetwork()
{
	// read in the name of the read and write directories from settings file
	struct stat buf;

	// if the file doesn't exist something is wrong so just set them to default settings
	if (stat(Options::FileIO::FILE_SETTINGS, &buf) == -1)
	{
		readDir = "bwapi-data/testio/read/";
		writeDir = "bwapi-data/testio/write/";
	}
	else
	{
		std::ifstream f_in(Options::FileIO::FILE_SETTINGS);
		getline(f_in, readDir);
		getline(f_in, writeDir);
		f_in.close();
	}

	// the file corresponding to the enemy's previous results
	std::string readFile = readDir + BWAPI::Broodwar->enemy()->getName() + ".net";

	// if the file doesn't exist, set the results to zeros
	if (stat(readFile.c_str(), &buf) == -1)
	{
		std::fill(results.begin(), results.end(), IntPair(0, 0));
	}
	// otherwise read in the results
	else
	{
		std::ifstream f_in(readFile.c_str());
		std::string line;
		getline(f_in, line);
		/*
		results[ProtossZealotRush].first = atoi(line.c_str());
		getline(f_in, line);
		results[ProtossZealotRush].second = atoi(line.c_str());
		getline(f_in, line);
		results[ProtossDarkTemplar].first = atoi(line.c_str());
		getline(f_in, line);
		results[ProtossDarkTemplar].second = atoi(line.c_str());
		getline(f_in, line);
		results[ProtossDragoons].first = atoi(line.c_str());
		getline(f_in, line);
		results[ProtossDragoons].second = atoi(line.c_str());
		*/
		f_in.close();
	}

//	BWAPI::Broodwar->printf("Results (%s): (%d %d) (%d %d) (%d %d)", BWAPI::Broodwar->enemy()->getName().c_str(),
	//	results[0].first, results[0].second, results[1].first, results[1].second, results[2].first, results[2].second);
}
/*
void Neural::writeNetwork()
{
//	std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ".txt";
//	std::ofstream f_out(writeFile.c_str());

//	f_out << results[ProtossZealotRush].first << "\n";
//	unsigned int decimal_point = net.save_to_fixed("D:/work/FANN-2.2.0-Source/bin/xor_fixed.net");
//	data.save_train_to_fixed("D:/work/FANN-2.2.0-Source/bin/xor_fixed.data", decimal_point);


//	f_out.close();
}
*/

void Neural::onEnd(const bool isWinner)
{
	// if the game ended before the tournament time limit
	if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
	{
		if (isWinner)
		{
			//	results[getCurrentStrategy()].first = results[getCurrentStrategy()].first + 1;

		}
		else
		{
			//results[getCurrentStrategy()].second = results[getCurrentStrategy()].second + 1;
		}
	}
	// otherwise game timed out so use in-game score
	else
	{
		if (getScore(BWAPI::Broodwar->self()) > getScore(BWAPI::Broodwar->enemy()))
		{
			//results[getCurrentStrategy()].first = results[getCurrentStrategy()].first + 1;
		}
		else
		{
			//	results[getCurrentStrategy()].second = results[getCurrentStrategy()].second + 1;
		}


	FANN::training_data data;


	data.set_train_data(3,num_input,(float**)&inputs[0],
									num_output,(float**)&outputs[0]);
	// Initialize and train the network with the data
	net.init_weights(data);
	net.train_on_data(data, max_iterations,
		iterations_between_reports, desired_error);

	net.save(writeDir + BWAPI::Broodwar->enemy()->getName() + ".net");
	}
}

// Test function that demonstrates usage of the fann C++ wrapper
void	Neural::createNetwork()
{
	//	cout << endl << "Creating network." << endl;
	if (net.create_from_file(readDir + BWAPI::Broodwar->enemy()->getName() + ".net")){

	}
	else
	{
		net.create_standard(num_layers, num_input, num_hidden, num_output);

		net.set_learning_rate(learning_rate);

		net.set_activation_steepness_hidden(1.0);
		net.set_activation_steepness_output(1.0);

		net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
		net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

		// Set additional properties such as the training algorithm
		net.set_training_algorithm(FANN::TRAIN_QUICKPROP);
	}
}
	

// Output network type and parameters
/*	cout << endl << "Network Type                         :  ";
	switch (net.get_network_type())
	{
	case FANN::LAYER:
		cout << "LAYER" << endl;
		break;
	case FANN::SHORTCUT:
		cout << "SHORTCUT" << endl;
		break;
	default:
		cout << "UNKNOWN" << endl;
		break;
	}
	*/
//	net.print_parameters();

//	cout << endl << "Training network." << endl;

/* Startup function. Syncronizes C and C++ output, calls the test function
and reports any exceptions*/
/*
int main(int argc, char **argv)
{
	try
	{
		std::ios::sync_with_stdio(); // Syncronize cout and printf output
		xor_test();
	}
	catch (...)
	{
		cerr << endl << "Abnormal exception." << endl;
	}
	return 0;
}
*/

const int Neural::getScore(BWAPI::Player * player) const
{
	return player->getBuildingScore() + player->getKillScore() + player->getRazingScore() + player->getUnitScore();
}

void Neural::setActions()
{
	






}

void Neural::selectBestAction(){

	fann_type *best_out;
	std::vector<float> 	actions(num_actions);
	std::vector<float>	states(num_states);
	std::vector<float>	input;
	for (int i; i < num_actions;++i){

		input.insert(actions.end(), states.begin(),
			states.end());
		fann_type *calc_out = net.run(&input[0]);
		if (best_out[0] < calc_out[0]){
			best_out[0] = calc_out[0];
			bestinput = input;
		}
	}
}


void Neural::addActions()
{
	protossOpeningBook = std::vector<std::string>(NumProtossStrategies);
	terranOpeningBook = std::vector<std::string>(NumTerranStrategies);
	zergOpeningBook = std::vector<std::string>(NumZergStrategies);

	//protossOpeningBook[ProtossZealotRush]	= "0 0 0 0 1 0 0 3 0 0 3 0 1 3 0 4 4 4 4 4 1 0 4 4 4";
	protossOpeningBook[ProtossZealotRush] = "0 0 0 0 1 0 3 3 0 0 4 1 4 4 0 4 4 0 1 4 3 0 1 0 4 0 4 4 4 4 1 0 4 4 4";
	//protossOpeningBook[ProtossZealotRush]	= "0";
	//protossOpeningBook[ProtossDarkTemplar]	= "0 0 0 0 1 3 0 7 5 0 0 12 3 13 0 22 22 22 22 0 1 0";
	protossOpeningBook[ProtossDarkTemplar] = "0 0 0 0 1 0 3 0 7 0 5 0 12 0 13 3 22 22 1 22 22 0 1 0";
	protossOpeningBook[ProtossDragoons] = "0 0 0 0 1 0 0 3 0 7 0 0 5 0 0 3 8 6 1 6 6 0 3 1 0 6 6 6";
	terranOpeningBook[TerranMarineRush] = "0 0 0 0 0 1 0 0 3 0 0 3 0 1 0 4 0 0 0 6";
	zergOpeningBook[ZergZerglingRush] = "0 0 0 0 0 1 0 0 0 2 3 5 0 0 0 0 0 0 1 6";

	if (selfRace == BWAPI::Races::Protoss)
	{
		results = std::vector<IntPair>(NumProtossStrategies);

		if (enemyRace == BWAPI::Races::Protoss)
		{
			usableStrategies.push_back(ProtossZealotRush);
			usableStrategies.push_back(ProtossDarkTemplar);
			usableStrategies.push_back(ProtossDragoons);
		}
		else if (enemyRace == BWAPI::Races::Terran)
		{
			usableStrategies.push_back(ProtossZealotRush);
			usableStrategies.push_back(ProtossDarkTemplar);
			usableStrategies.push_back(ProtossDragoons);
		}
		else if (enemyRace == BWAPI::Races::Zerg)
		{
			usableStrategies.push_back(ProtossZealotRush);
			usableStrategies.push_back(ProtossDragoons);
		}
		else
		{
			BWAPI::Broodwar->printf("Enemy Race Unknown");
			usableStrategies.push_back(ProtossZealotRush);
			usableStrategies.push_back(ProtossDragoons);
		}
	}
	else if (selfRace == BWAPI::Races::Terran)
	{
		results = std::vector<IntPair>(NumTerranStrategies);
		usableStrategies.push_back(TerranMarineRush);
	}
	else if (selfRace == BWAPI::Races::Zerg)
	{
		results = std::vector<IntPair>(NumZergStrategies);
		usableStrategies.push_back(ZergZerglingRush);
	}

	if (Options::Modules::USING_STRATEGY_IO)
	{
		readResults();
	}
}
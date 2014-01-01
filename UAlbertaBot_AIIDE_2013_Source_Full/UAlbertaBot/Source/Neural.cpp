#include "Common.h"
#include "Neural.h"
#include "Layer.h"
#include "NN.h"
#include "fann.h"
#include "fann_train.h"


// constructor
Neural::Neural()
:selfRace(BWAPI::Broodwar->self()->getRace())
,enemyRace(BWAPI::Broodwar->enemy()->getRace())
{
	addStrategies();
	setStrategy();
}

// get an instance of this
Neural & Neural::Instance()
{
	static Neural instance;
	return instance;
}

class Neural {

}


// read weight of Network

void Neural::readResults()
{
	// read in the name of the read and write directories from settings file
	struct stat buf;

	// if the file doesn't exist something is wrong so just set them to default settings
	if (stat(Options::FileIO::FILE_SETTINGS, &buf) == -1)
	{
		readNetwork = "bwapi-data/testio/read/";
		writeNetwork = "bwapi-data/testio/write/";
	}
	else
	{
		std::ifstream f_in(Options::FileIO::FILE_SETTINGS);
		getline(f_in, readDir);
		getline(f_in, writeDir);
		f_in.close();
	}

	// the file corresponding to the enemy's previous results
	std::string readFile = readNetwork + BWAPI::Broodwar->enemy()->getName() + "_network.txt";

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
		f_in.close();
	}

	BWAPI::Broodwar->printf("Results (%s): (%d %d) (%d %d) (%d %d)", BWAPI::Broodwar->enemy()->getName().c_str(),
		results[0].first, results[0].second, results[1].first, results[1].second, results[2].first, results[2].second);
}

void Neural::writeResults()
{
	std::string writeFile = writeDir + BWAPI::Broodwar->enemy()->getName() + ".txt";
	std::ofstream f_out(writeFile.c_str());

	f_out << results[ProtossZealotRush].first << "\n";
	f_out << results[ProtossZealotRush].second << "\n";
	f_out << results[ProtossDarkTemplar].first << "\n";
	f_out << results[ProtossDarkTemplar].second << "\n";
	f_out << results[ProtossDragoons].first << "\n";
	f_out << results[ProtossDragoons].second << "\n";

	f_out.close();
}

void Neural::onEnd(const bool isWinner)
{
	// write the win/loss data to file if we're using IO
	if (Options::Modules::USING_STRATEGY_IO)
	{
		// if the game ended before the tournament time limit
		if (BWAPI::Broodwar->getFrameCount() < Options::Tournament::GAME_END_FRAME)
		{
			if (isWinner)
			{
				results[getCurrentStrategy()].first = results[getCurrentStrategy()].first + 1;
			}
			else
			{
				results[getCurrentStrategy()].second = results[getCurrentStrategy()].second + 1;
			}
		}
		// otherwise game timed out so use in-game score
		else
		{
			if (getScore(BWAPI::Broodwar->self()) > getScore(BWAPI::Broodwar->enemy()))
			{
				results[getCurrentStrategy()].first = results[getCurrentStrategy()].first + 1;
			}
			else
			{
				results[getCurrentStrategy()].second = results[getCurrentStrategy()].second + 1;
			}
		}

		writeResults();
	}
}


#include "floatfann.h"
#include "fann_cpp.h"

#include <ios>
#include <iostream>
#include <iomanip>
using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::left;
using std::right;
using std::showpos;
using std::noshowpos;


// Callback function that simply prints the information to cout
int print_callback(FANN::neural_net &net, FANN::training_data &train,
	unsigned int max_epochs, unsigned int epochs_between_reports,
	float desired_error, unsigned int epochs, void *user_data)
{
	cout << "Epochs     " << setw(8) << epochs << ". "
		<< "Current Error: " << left << net.get_MSE() << right << endl;
	return 0;
}

// Test function that demonstrates usage of the fann C++ wrapper
void xor_test()
{
	cout << endl << "XOR test started." << endl;

	const float learning_rate = 0.7f;
	const unsigned int num_layers = 3;
	const unsigned int num_input = 2;
	const unsigned int num_hidden = 3;
	const unsigned int num_output = 1;
	const float desired_error = 0.001f;
	const unsigned int max_iterations = 300000;
	const unsigned int iterations_between_reports = 1000;

	cout << endl << "Creating network." << endl;

	FANN::neural_net net;
	net.create_standard(num_layers, num_input, num_hidden, num_output);

	net.set_learning_rate(learning_rate);

	net.set_activation_steepness_hidden(1.0);
	net.set_activation_steepness_output(1.0);

	net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

	// Set additional properties such as the training algorithm
	//net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

	// Output network type and parameters
	cout << endl << "Network Type                         :  ";
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
	net.print_parameters();

	cout << endl << "Training network." << endl;

	FANN::training_data data;



	if (data.read_train_from_file("D:/work/FANN-2.2.0-Source/bin/xor.data"))
	{
		// Initialize and train the network with the data
		net.init_weights(data);

		cout << "Max Epochs " << setw(8) << max_iterations << ". "
			<< "Desired Error: " << left << desired_error << right << endl;
		net.set_callback(print_callback, NULL);
		net.train_on_data(data, max_iterations,
			iterations_between_reports, desired_error);

		cout << endl << "Testing network." << endl;

		for (unsigned int i = 0; i < data.length_train_data(); ++i)
		{
			// Run the network on the test data
			fann_type *calc_out = net.run(data.get_input()[i]);

			cout << "XOR test (" << showpos << data.get_input()[i][0] << ", "
				<< data.get_input()[i][1] << ") -> " << *calc_out
				<< ", should be " << data.get_output()[i][0] << ", "
				<< "difference = " << noshowpos
				<< fann_abs(*calc_out - data.get_output()[i][0]) << endl;
		}

		cout << endl << "Saving network." << endl;

		// Save the network in floating point and fixed point
		net.save("D:/work/FANN-2.2.0-Source/bin/xor_float.net");
		unsigned int decimal_point = net.save_to_fixed("D:/work/FANN-2.2.0-Source/bin/xor_fixed.net");
		data.save_train_to_fixed("D:/work/FANN-2.2.0-Source/bin/xor_fixed.data", decimal_point);

		cout << endl << "XOR test completed." << endl;
	}
}

/* Startup function. Syncronizes C and C++ output, calls the test function
and reports any exceptions */
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

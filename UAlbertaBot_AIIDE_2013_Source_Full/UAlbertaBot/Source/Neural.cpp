#include "Common.h"
#include "Neural.h"
#include "Layer.h"
#include "NN.h"
#include "fann.h"
#include "fann_train.h"


// constructor
Neural::Neural()
: firstAttackSent(false)
, currentStrategy(0)
, selfRace(BWAPI::Broodwar->self()->getRace())
, enemyRace(BWAPI::Broodwar->enemy()->getRace())
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
	Layer::Instance();
	struct fann_train_data *data = fann_read_train_from_file("train.txt");
	fann_reset_MSE(ann);
	fann_test_data(ann, data);
	printf("Mean Square Error: %f\n", fann_get_MSE(ann));
	fann_destroy_train(data);


	nn.init(2, 2, 1);
	nn.setLearningRate(0.2);

	// 訓練データの作成
	double[][] trainingSet = new double[4][3];
	// 訓練データ0
	trainingSet[0][0] = 0;  // 入力1
	trainingSet[0][1] = 0;  // 入力2
	trainingSet[0][2] = 0;  // 教師

	// 訓練データ1
	trainingSet[1][0] = 0;
	trainingSet[1][1] = 1;
	trainingSet[1][2] = 1;

	// 訓練データ2
	trainingSet[2][0] = 1;
	trainingSet[2][1] = 0;
	trainingSet[2][2] = 1;

	// 訓練データ3
	trainingSet[3][0] = 1;
	trainingSet[3][1] = 1;
	trainingSet[3][2] = 0;

	// 訓練データを学習
	double error = 1.0;
	int count = 0;
	while ((error > 0.0001) && (count < 50000)) {
		error = 0;
		count++;
		// 4つの訓練データを誤差が小さくなるまで繰り返し学習
		for (int i = 0; i<4; i++) {
			// 入力層に値を設定
			nn.setInput(0, trainingSet[i][0]);
			nn.setInput(1, trainingSet[i][1]);
			// 教師信号を設定
			nn.setTeacherValue(0, trainingSet[i][2]);
			// 学習開始
			nn.feedForward();
			error += nn.calculateError();
			nn.backPropagate();
		}
		error /= 4.0;
		System.out.println(count + "\t" + error);
	}

	// 学習完了
	nn.setInput(0, 0);  // 入力1
	nn.setInput(1, 0);  // 入力2
	nn.feedForward();   // 出力を計算
	System.out.println(nn.getOutput(0));
}


// read weight of Network

void Neural::readResults()
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
	std::string readFile = readDir + BWAPI::Broodwar->enemy()->getName() + ".txt";

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


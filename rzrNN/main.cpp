#include "Network.h"
#include "ReadMNIST.h"
#include <thread>
#include <sstream>

#define NOMINMAX
#include <Windows.h>
inline void PrintImg(const std::vector<float>& _Image, const uint32_t _uRows, const uint32_t _uColumns)
{
	if (_Image.size() == _uRows * _uColumns)
	{
		for (uint32_t y = 0; y < _uColumns; ++y)
		{
			for (uint32_t x = 0; x < _uRows; ++x)
			{
				char chars[] = { ' ', (char)176,(char)177,(char)178,(char)219 };
				std::cout << chars[(uint32_t)(_Image[y*_uColumns + x] * sizeof(chars))];
			}
			std::cout << std::endl;
		}
	}
}

int main(int argc, char* argv[])
{
	//if (argc < 2)
	//	return 0;

	MNISTFile images("train-images.idx3-ubyte");
	MNISTFile labels("train-labels.idx1-ubyte");

	uint32_t uTrainingSize = std::min(50000u, images.GetCount());
	uint32_t uTestSize = std::min(images.GetCount() - uTrainingSize, 10000u);
	uint32_t uTotal = uTrainingSize + uTestSize;

	std::shared_ptr<Network> nn = nullptr;

	std::string line;
	std::cout << "Load model (name / no): ";
	std::getline(std::cin, line);

	bool bEval = false;

	if (line != "no")
	{
		nn = std::make_shared<Network>(line);

		std::cout << "Evaluate (yes/no): ";
		std::getline(std::cin, line);
		bEval = (line == "yes" || line == "y");
	}
	else
	{
		std::cout << "Enter layer sizes l1 l2 l3...: ";
		std::getline(std::cin, line);

		std::stringstream ss(line);

		std::vector<uint32_t> Layers = { images.GetColumns()*images.GetRows() }; // input

		std::cout << "Layers: " << Layers.front() << " ";
		while (ss.good())
		{
			uint32_t uLayer = 10u;
			ss >> uLayer;
			Layers.push_back(uLayer);
			std::cout << uLayer << " ";
		}

		Layers.push_back(10u); // output
		std::cout << Layers.back() << std::endl;

		nn = std::make_shared<Network>(Layers);
	}

	if (nn->HasLayer() == false)
	{
		std::cout << "Failed to initialize network" << std::endl;
		return 0;
	}

	std::vector<float> img, lbl;
	
	Layer& Input(nn->GetInputLayer());
	Layer& Output(nn->GetOutputLayer());

	if (bEval)
	{
		uint32_t uIndex = 0;
		while (uIndex != -1)
		{
			std::cout << "Enter MNIST index to classify: ";
			std::cin >> uIndex;

			images.GetImage(uIndex, img);
			PrintImg(img, images.GetRows(), images.GetColumns());
			Input.FeedForward(img);

			std::cout << "Predicted " << Output.ArgMax() << " [" << images.GetLabel(uIndex) << "]" << std::endl;
		}

		return 0;
	}

	char pConTitle[256];

	float fLearningRate = 0.5f;

	std::cout << "Enter learning rate: ";
	std::cin >> fLearningRate;

	static uint32_t uUpdate = 0u;

	for (uint32_t e = 0; e < 10; ++e)
	{
		uint32_t uCorrectTrain = 0u;
		uint32_t uCorrectTest = 0u;

		// train
		for (uint32_t i = 0; i < uTrainingSize; ++i)
		{
			images.GetImage(i, img);
			labels.GetLabelVector(i, lbl);

			Input.BackProp(img, lbl, fLearningRate, uTrainingSize);

			if (lbl[Output.ArgMax()] == 1.f)
			{
				++uCorrectTrain;
			}

			if (uUpdate++ == 500u)
			{
				snprintf(pConTitle, sizeof(pConTitle), "Epoch %d %d/%d %f\0", e, i + 1, uTotal, fLearningRate);
				SetConsoleTitleA(pConTitle);
				uUpdate = 0u;
			}
		}

		// validate
		for (uint32_t i = uTrainingSize; i < uTotal; ++i)
		{
			images.GetImage(i, img);
			labels.GetLabelVector(i, lbl);

			Input.FeedForward(img);
			
			if (lbl[Output.ArgMax()] == 1.f)
			{
				++uCorrectTest;
			}

			if (uUpdate++ == 2000u)
			{
				snprintf(pConTitle, sizeof(pConTitle), "Epoch %d %d/%d %f\0", e, i + 1, uTotal, fLearningRate);
				SetConsoleTitleA(pConTitle);
				uUpdate = 0u;
			}
		}

		float fTrain = (float)uCorrectTrain*100.f / (float)uTrainingSize;
		float fTest = (float)uCorrectTest*100.f / (float)uTestSize;

		std::cout << "Epoch " << e << ": " 
			<< uCorrectTrain << "/" << uTrainingSize << "\t"
			<< uCorrectTest << "/" << uTestSize << "\t" << fTrain << "\t" << fTest << std::endl;
	}

	nn->Save("model.rzrnn");

	//uint32_t uIndex = 0;
	//while (uIndex != -1)
	//{
	//	std::cin >> uIndex;
	//	if (mnist.GetType() == kImages)
	//	{
	//		mnist.GetImage(uIndex, Image);
	//		PrintImg(Image, mnist.GetRows(), mnist.GetColumns());
	//	}
	//	else if (mnist.GetType() == kLabels)
	//	{
	//		std::cout << "Label " << mnist.GetLabel(uIndex) << std::endl;
	//	}
	//}

	//for (uint32_t i = 0; i < mnist.GetCount(); i++)
	//{
	//	if (mnist.GetType() == kImages)
	//	{
	//		mnist.GetImage(i, Image);
	//		PrintImg(Image, mnist.GetRows(), mnist.GetColumns());
	//	}
	//	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	//	system("cls");
	//}

	system("pause");
	return 0;
}

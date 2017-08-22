#pragma once
#include "Neuron.h"
#include "ReadMNIST.h"
#include <random>
#include <string>

class Network
{
public:
	Network(const std::vector<uint32_t>& _LayerSizes);
	Network(const std::string& _sModelPath);

	~Network() {};

	Layer& GetInputLayer() { return m_Layers.front(); }
	Layer& GetOutputLayer() { return m_Layers.back(); }

	void Save(const std::string& _sModelPath);
	void Create(const std::vector<uint32_t>& _LayerSizes);
	void InitNormalDist();

	bool HasLayer() const { return m_Layers.empty() == false; }

private:
	std::vector<Layer> m_Layers;
};

inline Network::Network(const std::vector<uint32_t>& _LayerSizes)
{
	Create(_LayerSizes);
	InitNormalDist();
}

inline Network::Network(const std::string & _sModelPath)
{
	std::ifstream model(_sModelPath.c_str(), std::ios_base::binary);

	if (model.is_open() == false)
		return;

	std::cout << "Loading " << _sModelPath << "..." << std::endl;

	uint32_t uLayers = 0u;
	model >> uLayers;

	std::vector<uint32_t> LayerSizes;

	for (uint32_t i = 0; i < uLayers; i++)
	{
		uint32_t uSize = 0u;
		model >> uSize;
		LayerSizes.push_back(uSize);
	}

	Create(LayerSizes);

	uint32_t uIdx = 0;
	for (Layer& l : m_Layers)
	{
		uint32_t uWeights = uIdx > 0u ? LayerSizes[uIdx - 1u] : 0u;
		for (Neuron& n : l.m_Neurons)
		{
			model >> n.m_fBias;
			if (uWeights > 0u)
			{
				n.m_Weights.resize(uWeights);
				for (float& w : n.m_Weights)
				{
					model >> w;
				}
			}
		}
		++uIdx;
	}
}

inline void Network::Save(const std::string& _sModelPath)
{
	std::ofstream model(_sModelPath.c_str(), std::ios_base::binary);

	if (model.is_open() == false)
		return;

	std::cout << "Saving " << _sModelPath << "..." << std::endl;

	model << (uint32_t)m_Layers.size();

	for (const Layer& l : m_Layers)
	{
		model << (uint32_t)l.m_Neurons.size();
	}

	for (const Layer& l : m_Layers)
	{
		for (const Neuron& n : l.m_Neurons)
		{
			model << n.m_fBias;
			for (const float& w : n.m_Weights)
			{
				model << w;
			}
		}
	}

	model.close();
}

inline void Network::Create(const std::vector<uint32_t>& _LayerSizes)
{
	if (_LayerSizes.empty())
		return;

	for (const uint32_t& neurons : _LayerSizes)
	{
		m_Layers.emplace_back(neurons);
	}

	for (auto it = m_Layers.begin(); it != m_Layers.end(); ++it)
	{
		if (it != m_Layers.begin())
		{
			it->SetPrev(&*(it - 1));
		}

		if (it + 1 != m_Layers.end())
		{
			it->SetNext(&*(it + 1));
		}
	}

	// fully connected layers
	for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); ++it)
	{
		for (Neuron& n : it->m_Neurons)
		{
			if (it + 1 != m_Layers.rend())
			{
				size_t uInputs = (it + 1)->m_Neurons.size();
				for (uint32_t i = 0; i < uInputs; i++)
				{
					n.m_InputIndices.push_back(i);
				}
			}
		}
	}
}

inline void Network::InitNormalDist()
{
	std::default_random_engine gen1, gen2;
	std::normal_distribution<float> distribution(0.f, 1.f);

	// fully connected layers
	for (auto it = m_Layers.rbegin(); it != m_Layers.rend(); ++it)
	{
		for (Neuron& n : it->m_Neurons)
		{
			if (it + 1 != m_Layers.rend())
			{
				size_t uInputs = (it + 1)->m_Neurons.size();
				for (uint32_t i = 0; i < uInputs; i++)
				{
					n.m_Weights.push_back(distribution(gen1));
					n.m_fBias = distribution(gen2);
				}
			}
		}
	}
}
